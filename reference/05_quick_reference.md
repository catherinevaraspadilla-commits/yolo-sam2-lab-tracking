# Quick Reference — Cheatsheet

Referencia rápida para aplicar los componentes en otro programa.

---

## Importaciones

```python
# Pipeline completo
from core.pipeline import ProcessMedia
from core.yolo_processor import YOLOProcessor
from core.sam_processor import SAM2Processor
from core.visualization import VisualizationProcessor

# Tracking standalone
from tracking.identity_matcher import IdentityMatcher
from tracking.mask_utils import calculate_iou, filter_duplicates
from tracking.trajectory_tracker import TrajectoryTracker

# Modelos directos
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
```

---

## Parámetros de YOLOProcessor

| Parámetro | Tipo | Default | Uso |
|-----------|------|---------|-----|
| `model` | `str` | `"yolov8n.pt"` | Ruta al modelo |
| `confidence` | `float` | `0.5` | Umbral de confianza |
| `output_key` | `str` | `"yolo"` | Namespace en frame_data |
| `entities` | `int\|None` | `None` | Mantener exactamente N |
| `max_entities` | `int\|None` | `None` | Mantener máximo N |
| `min_entities` | `int\|None` | `None` | Mínimo N (reintenta bajando umbral) |
| `area_min` | `int\|None` | `None` | Área mínima (px²) |
| `area_max` | `int\|None` | `None` | Área máxima (px²) |
| `area` | `float\|None` | `None` | Área adaptativa (del 1er frame) |
| `area_error` | `float` | `0.2` | Tolerancia del área adaptativa |
| `classes` | `list\|None` | `None` | IDs de clase a incluir |
| `exclude_classes` | `list\|None` | `None` | IDs de clase a excluir |
| `max_overlap` | `float\|None` | `None` | Umbral NMS (IoU) |
| `edge_margin` | `int` | `0` | Margen de borde en px |
| `roi` | `list\|None` | `None` | Región `[x1, y1, x2, y2]` |

---

## Parámetros de SAM2Processor

| Parámetro | Tipo | Default | Uso |
|-----------|------|---------|-----|
| `model_type` | `str` | `"large"` | `"large"` o `"tiny"` |
| `input_source` | `str\|list` | requerido | `"namespace:key"` o lista de prompts |
| `output_key` | `str` | `"sam2"` | Namespace en frame_data |
| `max_entities` | `int` | `2` | Número fijo de slots de identidad |
| `proximity_threshold` | `float` | `50` | Distancia máx. entre frames (px) |
| `area_tolerance` | `float` | `0.4` | Variación de área permitida (±40%) |
| `iou_threshold` | `float` | `0.5` | IoU para deduplicar máscaras |

---

## Parámetros de VisualizationProcessor

| Parámetro | Tipo | Default | Uso |
|-----------|------|---------|-----|
| `input_keys` | `dict` | `{}` | `{"ns": ["key1", "key2"]}` |
| `show_masks` | `bool` | `False` | Overlay de máscara |
| `show_boxes` | `bool` | `False` | Bounding boxes |
| `show_trajectories` | `bool` | `False` | Trayectorias históricas |
| `show_keypoints` | `bool` | `False` | Keypoints |
| `show_centroids` | `bool` | `True` | Puntos en centroides |
| `trail_length` | `int` | `30` | Frames de historial |
| `mask_alpha` | `float` | `0.35` | Transparencia máscara (0–1) |
| `box_thickness` | `int` | `2` | Grosor del box |
| `font_scale` | `float` | `0.6` | Tamaño del texto |
| `mask_border_thickness` | `int` | `2` | Grosor del borde de máscara |
| `trail_thickness` | `int` | `2` | Grosor de la trayectoria |

---

## Formatos de datos

### Entrada a process()

```python
frame_data = {
    'frame':        np.ndarray,  # BGR, shape (H, W, 3), dtype uint8
    'frame_index':  int,
    'metadata':     dict,
    'previous_data': dict | None
}
```

### Salida de YOLOProcessor

```python
frame_data['yolo'] = {
    'boxes':       np.ndarray,  # [N, 4] — float, [x1, y1, x2, y2] px absolutos
    'classes':     np.ndarray,  # [N]    — int, índice de clase
    'labels':      list[str],   # [N]    — nombre de clase
    'confidences': np.ndarray,  # [N]    — float 0.0–1.0
    'masks':       list,        # [N]    — arrays o lista vacía
    'keypoints':   list,        # [N]    — arrays o lista vacía
}
```

### Salida de SAM2Processor

```python
frame_data['sam2'] = {
    'masks':     list[np.ndarray],         # [max_entities] — bool, (H, W) o zeros
    'centroids': list[tuple | None],       # [max_entities] — (x, y) o None
    'areas':     list[float],              # [max_entities] — px² o 0
    'scores':    list[float],              # [max_entities] — 0.0–1.0 o 0.0
}
```

---

## Snippets de uso frecuente

### Leer detecciones YOLO

```python
yolo_out = frame_data.get('yolo', {})
boxes       = yolo_out.get('boxes', [])
labels      = yolo_out.get('labels', [])
confidences = yolo_out.get('confidences', [])

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)
    label = labels[i] if i < len(labels) else "?"
    conf  = confidences[i] if i < len(confidences) else 0
    print(f"  [{i}] {label} ({conf:.2f}) @ [{x1},{y1},{x2},{y2}]")
```

### Leer máscaras SAM2

```python
sam_out = frame_data.get('sam2', {})
masks     = sam_out.get('masks', [])
centroids = sam_out.get('centroids', [])

for slot_idx, (mask, centroid) in enumerate(zip(masks, centroids)):
    if mask is None or centroid is None:
        print(f"  Slot {slot_idx}: inactivo")
        continue
    cx, cy = int(centroid[0]), int(centroid[1])
    area   = mask.sum()
    print(f"  Slot {slot_idx}: centroide=({cx},{cy}), área={area}px²")
```

### Calcular bounding box de una máscara SAM2

```python
import numpy as np

def mask_to_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax, rmax]  # [x1, y1, x2, y2]
```

### Calcular IoU entre dos máscaras

```python
from tracking.mask_utils import calculate_iou

iou = calculate_iou(mask_a, mask_b)  # 0.0–1.0
```

### Aplicar overlay de máscara manualmente

```python
import cv2, numpy as np

def draw_mask(frame, mask, color=(0, 255, 0), alpha=0.4):
    overlay = frame.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
```

### Inferencia YOLO directa (una línea)

```python
from ultralytics import YOLO
results = YOLO("yolov8n.pt")(frame, conf=0.5, verbose=False)[0]
boxes = results.boxes.xyxy.cpu().numpy()  # [N, 4]
```

### Inferencia SAM2 directa

```python
predictor.set_image(frame)
masks, scores, _ = predictor.predict(
    box=np.array([x1, y1, x2, y2]),
    multimask_output=False
)
mask = masks[0]  # np.ndarray bool [H, W]
```

---

## Estructura de salida JSON

```json
{
  "info": {
    "source": "video.mp4",
    "type": "video",
    "resolution": "1920x1080",
    "fps": 30,
    "total_frames": 300,
    "timestamp": "20250224_153045",
    "layers": ["YOLOProcessor", "SAM2Processor", "VisualizationProcessor"]
  },
  "results": [
    {
      "frame_index": 0,
      "metadata": {
        "yolo_info": { "model": "yolov8n.pt", "detections_count": 2 },
        "sam2_info": { "entities_active": 2, "max_entities": 2, "mode": "bus" }
      },
      "yolo": {
        "boxes": [[100, 50, 300, 200]],
        "classes": [0],
        "labels": ["rat"],
        "confidences": [0.92],
        "masks": [],
        "keypoints": []
      },
      "sam2": {
        "masks": ["<array serializada>"],
        "centroids": [[200, 125]],
        "areas": [15000],
        "scores": [0.96]
      }
    }
  ]
}
```

---

## Configuraciones recomendadas por caso de uso

### Seguimiento de 1–2 animales en laboratorio
```python
YOLOProcessor(model="mi_modelo.pt", confidence=0.6, entities=2, edge_margin=20)
SAM2Processor(input_source="yolo:boxes", max_entities=2, proximity_threshold=60)
```

### Detección de personas en cámara fija con ROI
```python
YOLOProcessor(model="yolov8n.pt", confidence=0.5, classes=[0],
              roi=[200, 100, 1200, 900], max_entities=10)
```

### Segmentación sin detector previo (inicialización manual)
```python
SAM2Processor(
    input_source=[{"points": [[x, y]], "labels": [1]} for x, y in puntos_iniciales],
    max_entities=len(puntos_iniciales),
    proximity_threshold=100,
    area_tolerance=0.6
)
```

### Pipeline mínimo sin visualización
```python
ProcessMedia(
    source="video.mp4",
    processors=[
        YOLOProcessor(model="yolov8n.pt", confidence=0.5),
        SAM2Processor(input_source="yolo:boxes", max_entities=3)
    ],
    output="json"   # Solo exportar JSON, sin renderizar video
)
```
