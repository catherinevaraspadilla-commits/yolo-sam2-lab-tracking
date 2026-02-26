# Comparacion de Pipelines

Este proyecto tiene 3 pipelines para segmentacion y tracking de ratas.
Todos usan YOLO (deteccion + keypoints) y SAM2 (segmentacion), pero
difieren en como manejan la identidad y que pasa cuando YOLO falla.

---

## Tabla Comparativa

| Aspecto | sam2_yolo | sam2_video | reference |
|---------|-----------|------------|-----------|
| **SAM2** | ImagePredictor (stateless) | VideoPredictor (temporal memory) | ImagePredictor (stateless) |
| **YOLO** | detect + track (BoT-SORT) | detect-only | detect-only |
| **Tracking** | SlotTracker (YOLO IDs + Hungarian) | SAM2 temporal memory | IdentityMatcher (centroid + area) |
| **Centroid fallback** | No | Si (SAM2 temporal) | Si (punto SAM2 desde centroide previo) |
| **Batch/chunks** | Si | No | Si |
| **Contacts** | Si | Si | Si |
| **Velocidad** | Rapida (~25 FPS) | Lenta (~5-10 FPS) | Rapida (~25 FPS) |
| **Robustez oclusion** | Media | Alta | Media-Alta |

---

## 1. sam2_yolo

**Modulo:** `src/pipelines/sam2_yolo/`
**Configs:** `configs/local_quick.yaml`, `configs/hpc_full.yaml`
**Slurm:** `slurm/run_infer.sbatch`, `slurm/run_chunks.sbatch`

### Flujo por frame

```
Frame → YOLO.track() (BoT-SORT) → boxes + track_ids + keypoints
      → SAM2ImagePredictor.predict(box=...) → mascaras
      → SlotTracker (YOLO IDs + Hungarian) → slots fijos
      → ContactTracker (opcional)
      → Render overlay
```

### Como funciona el tracking

1. YOLO usa BoT-SORT internamente → da track_id a cada deteccion
2. SlotTracker intenta mapear track_ids a slots fijos (slot 0 = verde, slot 1 = rojo)
3. Si YOLO cambia IDs, un "swap guard" compara costos straight vs swapped
4. Fallback: Hungarian assignment con costo suave (distancia + IoU + area)

### Soporte batch

```bash
# Job array con 4 chunks paralelos
sbatch --array=0-3 slurm/run_chunks.sbatch

# Cada chunk procesa un rango de frames
python -m src.pipelines.sam2_yolo.run --config configs/hpc_full.yaml \
    --start-frame 0 --end-frame 750 --chunk-id 0

# Merge al final
python scripts/merge_chunks.py outputs/runs/*_chunk*/ -o outputs/runs/merged
```

### Debilidades

- Depende de YOLO cada frame. Si YOLO pierde una rata → slot vacio, la rata "desaparece"
- BoT-SORT puede hacer ID switch durante cruces/interacciones
- No hay fallback: si YOLO no detecta, SAM2 no tiene prompt

### Cuando usar

- Videos sin muchas interacciones entre ratas
- Cuando necesitas batch processing (videos largos)
- Cuando la velocidad importa

---

## 2. sam2_video

**Modulo:** `src/pipelines/sam2_video/`
**Configs:** `configs/local_sam2video.yaml`, `configs/hpc_sam2video.yaml`
**Slurm:** `slurm/run_sam2video.sbatch`

### Flujo por segmento (2000 frames)

```
Segmento:
  1. Extraer frames a JPEGs temporales
  2. YOLO detect_only() en primer frame → boxes para init
  3. SAM2VideoPredictor.init_state(frames_dir)
  4. SAM2.add_new_points_or_box(frame_idx=0, box=...) por cada rata
  5. SAM2.propagate_in_video() → genera mascaras para todos los frames
     Por cada frame:
       a. YOLO detect_only() → keypoints
       b. Match keypoints ↔ mascaras SAM2 (nose-in-mask o centroide)
       c. ContactTracker (opcional)
       d. Render overlay
  6. Limpiar JPEGs + GPU memory
```

### Como funciona el tracking

- SAM2VideoPredictor tiene memory bank de 7 frames (memory_attention + memory_encoder)
- Una vez inicializado con boxes, propaga mascaras automaticamente
- No necesita YOLO en frames intermedios (solo para keypoints)
- Identidad mantenida por SAM2 internamente (object pointers)
- Entre segmentos: matching de centroides para preservar colores

### Soporte batch

**No tiene soporte batch.** Procesa el video completo de una vez (dividido en
segmentos internos de ~2000 frames para manejar memoria, pero no se puede
paralelizar entre GPUs).

### Debilidades

- Lento: ~3-5x mas lento que los otros pipelines
- Requiere extraer frames a JPEG (I/O adicional)
- Mas uso de memoria (GPU + RAM para frames)
- No se puede paralelizar con chunks

### Cuando usar

- Videos cortos con muchas interacciones/oclusiones
- Cuando la calidad del tracking importa mas que la velocidad
- Cuando otros pipelines fallan en frames especificos

---

## 3. reference

**Modulo:** `src/pipelines/reference/`
**Configs:** `configs/local_reference.yaml`, `configs/hpc_reference.yaml`
**Slurm:** `slurm/run_reference.sbatch`

### Flujo por frame

```
Frame → YOLO detect_only() → boxes + keypoints (SIN BoT-SORT)
      → SAM2ImagePredictor.set_image(frame)
      → Para cada deteccion: SAM2.predict(box=...) → mascara
      → Si YOLO detecto menos ratas que slots activos:
          Para cada slot sin match:
            SAM2.predict(point_coords=prev_centroid) → mascara fallback
      → filter_duplicates() → eliminar mascaras redundantes
      → IdentityMatcher.match() → asignar a slots fijos
      → ContactTracker (opcional)
      → Render overlay
```

### Como funciona el tracking

1. IdentityMatcher tiene slots fijos con centroide y area del frame anterior
2. Para cada frame, calcula distancia entre centroides actuales y previos
3. Asigna mascara mas cercana a cada slot si:
   - Distancia < proximity_threshold (default 60px)
   - Variacion de area < area_tolerance (default ±40%)
4. Mascaras sin slot → llenan slots vacios

### Centroid fallback (pieza clave)

Cuando YOLO detecta solo 1 rata pero IdentityMatcher sabe que hay 2 slots activos:

```python
# El slot perdido tiene un centroide previo
prev_centroid = matcher.prev_centroids[slot_idx]

# Se usa como punto prompt para SAM2
SAM2.predict(point_coords=prev_centroid, point_labels=[1])
```

Esto recupera la mascara de la rata perdida usando SAM2 con un punto
en vez de un box. Es lo que hace que el pipeline funcione durante oclusiones.

### Soporte batch

```bash
# Soporta los mismos args que sam2_yolo
python -m src.pipelines.reference.run --config configs/hpc_reference.yaml \
    --start-frame 0 --end-frame 750 --chunk-id 0

# Se puede usar con run_chunks.sbatch cambiando el modulo:
# python -m src.pipelines.reference.run en vez de src.pipelines.sam2_yolo.run
```

### Debilidades

- Sin memoria temporal de SAM2 → identidad puede flipear con movimiento rapido
- proximity_threshold necesita tuning segun resolucion
- Centroid fallback depende de que el centroide previo sea cercano a la posicion actual

### Cuando usar

- Videos con interacciones donde YOLO pierde ratas momentaneamente
- Cuando necesitas batch processing + robustez
- Como alternativa directa a sam2_yolo cuando hay ID switches

---

## Comparacion de Componentes

### Modelos

| Componente | sam2_yolo | sam2_video | reference |
|-----------|-----------|------------|-----------|
| YOLO | `model.track()` | `model()` detect-only | `model()` detect-only |
| SAM2 | `SAM2ImagePredictor` | `SAM2VideoPredictor` | `SAM2ImagePredictor` |
| SAM2 build | `build_sam2()` | `build_sam2_video_predictor()` | `build_sam2()` |

### Tracking / Identidad

| Componente | sam2_yolo | sam2_video | reference |
|-----------|-----------|------------|-----------|
| Clase | `SlotTracker` | SAM2 interno | `IdentityMatcher` |
| Archivo | `src/common/tracking.py` | N/A | `src/pipelines/reference/identity_matcher.py` |
| Identidad basada en | YOLO track IDs + Hungarian | SAM2 object pointers | Centroid distance + area |
| Usa YOLO track IDs | Si (primario) | No | No |
| Missing frames | 5 frames tolerancia | N/A (SAM2 propaga) | Mantiene prev_centroids |
| Swap guard | Si | N/A | No (no necesario sin BoT-SORT) |

### Archivos de Configuracion

| Pipeline | Local | HPC | Slurm |
|----------|-------|-----|-------|
| sam2_yolo | `configs/local_quick.yaml` | `configs/hpc_full.yaml` | `slurm/run_infer.sbatch`, `slurm/run_chunks.sbatch` |
| sam2_video | `configs/local_sam2video.yaml` | `configs/hpc_sam2video.yaml` | `slurm/run_sam2video.sbatch` |
| reference | `configs/local_reference.yaml` | `configs/hpc_reference.yaml` | `slurm/run_reference.sbatch` |

### Modulos Compartidos

Todos los pipelines reutilizan estos modulos comunes:

| Modulo | Archivo | Funcion |
|--------|---------|---------|
| Video I/O | `src/common/io_video.py` | Leer/escribir video, iterar frames |
| Config | `src/common/config_loader.py` | YAML + CLI overrides |
| Metricas | `src/common/metrics.py` | mask_iou, compute_centroid |
| Visualizacion | `src/common/visualization.py` | Overlays, keypoints, centroides |
| Contactos | `src/common/contacts.py` | ContactTracker completo |
| Estructuras | `src/common/utils.py` | Detection, Keypoint dataclasses |

---

## Comandos Rapidos

### Test local (clip de 10s)

```bash
# sam2_yolo
python -m src.pipelines.sam2_yolo.run --config configs/local_quick.yaml

# sam2_video
python -m src.pipelines.sam2_video.run --config configs/local_sam2video.yaml

# reference
python -m src.pipelines.reference.run --config configs/local_reference.yaml
```

### Bunya HPC (video de 2 min)

```bash
# sam2_yolo
sbatch --export=ALL,CONFIG=configs/hpc_full.yaml,OVERRIDES="video_path=data/raw/original_120s.avi" slurm/run_infer.sbatch

# sam2_video
sbatch --export=ALL,CONFIG=configs/hpc_sam2video.yaml,OVERRIDES="video_path=data/raw/original_120s.avi" slurm/run_sam2video.sbatch

# reference
sbatch --export=ALL,CONFIG=configs/hpc_reference.yaml,OVERRIDES="video_path=data/raw/original_120s.avi" slurm/run_reference.sbatch
```

### Bunya HPC con contacts

```bash
# Agregar contacts.enabled=true a OVERRIDES
sbatch --export=ALL,CONFIG=configs/hpc_reference.yaml,OVERRIDES="video_path=data/raw/original_120s.avi contacts.enabled=true" slurm/run_reference.sbatch
```

### Batch processing (solo sam2_yolo y reference)

```bash
# 4 chunks paralelos
sbatch --array=0-3 --export=ALL,CONFIG=configs/hpc_full.yaml slurm/run_chunks.sbatch

# Merge resultados
python scripts/merge_chunks.py outputs/runs/*_chunk*/ -o outputs/runs/merged
```

---

## Origen: Pipeline de Referencia

La carpeta `reference/` contiene la documentacion del toolkit original
(YOLOv8 + SAM2) que inspiro este proyecto. Sus componentes clave:

- **ProcessMedia**: Orquestador con patron Data Bus (frame_data dict)
- **YOLOProcessor**: Deteccion con filtros (area, ROI, edge_margin)
- **SAM2Processor**: Segmentacion con IdentityMatcher
- **VisualizationProcessor**: Renderizado modular

El pipeline `reference` de este proyecto porta la logica de IdentityMatcher
y centroid fallback del toolkit original, combinandola con nuestros YOLO
keypoints (7 puntos pose de rata) y contact classification.
