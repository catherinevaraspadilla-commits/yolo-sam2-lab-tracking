# Arquitectura General del Pipeline

## Estructura de archivos

```
yolov8_sam2_toolkit/
├── core/
│   ├── pipeline.py           # ProcessMedia — orquestador principal
│   ├── yolo_processor.py     # YOLOProcessor — detección de objetos
│   ├── sam_processor.py      # SAM2Processor — segmentación de instancias
│   ├── visualization.py      # VisualizationProcessor — renderizado
│   └── ExampleOfLayer.py     # Plantilla para capas personalizadas
├── tracking/
│   ├── identity_matcher.py   # IdentityMatcher — slots de identidad fijos
│   ├── mask_utils.py         # IoU y deduplicación de máscaras
│   └── trajectory_tracker.py # TrajectoryTracker — historial de posiciones
├── data/
│   ├── input/                # Videos e imágenes de entrada
│   └── output/               # Resultados generados
└── models/                   # Pesos de los modelos (.pt)
```

---

## Patrón de datos: Data Bus

El sistema usa un **diccionario compartido** (`frame_data`) que fluye a través de todos los procesadores. Cada procesador:

1. **Lee** datos de procesadores anteriores usando su clave de namespace
2. **Escribe** sus resultados bajo su propia clave
3. **Devuelve** el diccionario actualizado

### Estructura del frame_data

```python
frame_data = {
    # Datos base (inmutables, provistos por el pipeline)
    'frame':        np.ndarray,      # Imagen BGR de OpenCV (H, W, 3)
    'frame_index':  int,             # Número de frame (0-based)
    'metadata':     dict,            # Metadatos para el JSON de salida
    'previous_data': dict | None,    # frame_data del frame anterior

    # Escritos por YOLOProcessor (output_key="yolo")
    'yolo': {
        'boxes':       np.ndarray,   # Shape [N, 4] — x1, y1, x2, y2
        'classes':     np.ndarray,   # Shape [N]   — índices de clase
        'labels':      list[str],    # Nombres de clase ["rat", "person", ...]
        'confidences': np.ndarray,   # Shape [N]   — scores 0.0–1.0
        'masks':       list,         # Máscaras YOLO (si el modelo las provee)
        'keypoints':   list,         # Keypoints (si el modelo los provee)
    },

    # Escritos por SAM2Processor (output_key="sam2")
    'sam2': {
        'masks':      list[np.ndarray],        # [max_entities] máscaras binarias (H, W)
        'centroids':  list[tuple[float,float]],# [max_entities] centros (x, y)
        'areas':      list[float],             # [max_entities] área en píxeles²
        'scores':     list[float],             # [max_entities] confidence 0.0–1.0
    },

    # Escrito por VisualizationProcessor
    'vis_frame':    np.ndarray,      # Frame renderizado con overlay
}
```

> **Nota:** Los slots inactivos en las listas de SAM2 contienen `None` (centroid/mask) o `0`/`0.0` (area/score).

---

## Flujo de ejecución

```
ProcessMedia.run()
    │
    ├── Abre fuente (cv2.VideoCapture o imagen)
    │
    └── Por cada frame:
            │
            ├── _process_frame_logic(frame, frame_index)
            │       │
            │       ├── frame_data = {'frame': frame, 'frame_index': N, ...}
            │       │
            │       └── for processor in processors:
            │               frame_data = processor.process(frame_data)
            │
            └── _save_outputs(frame_data)
                    ├── video_writer.write(frame_data['vis_frame'])
                    ├── json_data['results'].append(serialized)
                    └── cv2.imwrite(frame_path, frame_data['vis_frame'])
```

### Código del loop principal (pipeline.py)

```python
# core/pipeline.py — Lines 77-97
def _process_frame_logic(self, frame, frame_count, previous_data=None):
    frame_data = {
        'frame': frame,
        'frame_index': frame_count,
        'metadata': {},
        'previous_data': previous_data
    }
    for processor in self.processors:
        result = processor.process(frame_data)
        if not isinstance(result, dict):
            raise TypeError("Processor must return a dict")
        frame_data = result
    return frame_data
```

---

## Contrato de un Procesador

Todo procesador debe cumplir:

```python
class MiProcesador:
    output_key = "mi_namespace"  # Clave bajo la que escribe en frame_data

    def process(self, frame_data: dict) -> dict:
        # 1. Leer datos del bus
        frame = frame_data['frame']

        # 2. Procesar
        resultado = ...

        # 3. Escribir en el bus (NO borrar claves existentes)
        frame_data[self.output_key] = resultado

        # 4. SIEMPRE devolver el diccionario actualizado
        return frame_data
```

### Método opcional `validate`

```python
def validate(self, previous_processors: list):
    """Verificar compatibilidad antes de procesar"""
    has_yolo = any(isinstance(p, YOLOProcessor) for p in previous_processors)
    if not has_yolo:
        raise ValueError("Este procesador requiere YOLOProcessor antes")
```

---

## Salidas del sistema

### Video (MP4)
```python
# Escribe frame_data['vis_frame'] si existe, sino frame_data['frame']
output_path = output_folder / f"{base_name}.mp4"
```

### JSON
```json
{
  "info": {
    "source": "data/input/video.mp4",
    "type": "video",
    "resolution": "1920x1080",
    "fps": 30,
    "total_frames": 300,
    "layers": ["YOLOProcessor", "SAM2Processor", "VisualizationProcessor"]
  },
  "results": [
    {
      "frame_index": 0,
      "metadata": { "yolo_info": {...}, "sam2_info": {...} },
      "yolo": { "boxes": [...], "classes": [...], ... },
      "sam2": { "masks": [...], "centroids": [...], ... }
    }
  ]
}
```

### Frames individuales
```
output/
  └── 20250224_153045/
        ├── video.mp4
        ├── output.json
        └── frames/
              ├── frame_00000.jpg
              ├── frame_00001.jpg
              └── ...
```
