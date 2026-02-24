# Integración de SAM2 (Segment Anything Model 2)

**Archivo fuente:** [core/sam_processor.py](../core/sam_processor.py)
**Tracking:** [tracking/identity_matcher.py](../tracking/identity_matcher.py), [tracking/mask_utils.py](../tracking/mask_utils.py), [tracking/trajectory_tracker.py](../tracking/trajectory_tracker.py)

---

## Carga del modelo

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pathlib import Path

SAM2_DIR = Path("ruta/a/sam2")

# Modelos disponibles: "large" o "tiny"
model_type = "large"
ckpt_path   = SAM2_DIR / f"checkpoints/sam2.1_hiera_{model_type}.pt"
config_name = f"configs/sam2.1/sam2.1_hiera_{'l' if model_type == 'large' else 't'}.yaml"

model     = build_sam2(config_name, str(ckpt_path), device="cuda")
predictor = SAM2ImagePredictor(model)
```

---

## Inferencia directa (sin pipeline)

SAM2 requiere tres pasos por frame:

```python
import numpy as np

# 1. Establecer la imagen (una vez por frame)
predictor.set_image(frame)  # frame: np.ndarray BGR de OpenCV (H, W, 3)

# 2a. Segmentar con bounding box
box = np.array([x1, y1, x2, y2])  # coordenadas en píxeles
masks, scores, logits = predictor.predict(
    box=box,
    multimask_output=False  # False → 1 máscara por input; True → 3 candidatos
)
# masks: np.ndarray [1, H, W] o [3, H, W] — booleano
# scores: np.ndarray [1] o [3] — confianza

# 2b. Segmentar con puntos
points = np.array([[cx, cy]])     # [N, 2] — coordenadas (x, y)
labels = np.array([1])            # [N]    — 1=foreground, 0=background
masks, scores, logits = predictor.predict(
    point_coords=points,
    point_labels=labels,
    multimask_output=False
)

# 3. Usar la máscara
mask = masks[0]  # np.ndarray [H, W] — booleano
```

---

## Clase SAM2Processor

### Constructor

```python
from core.sam_processor import SAM2Processor

sam = SAM2Processor(
    model_type          = "large",      # "large" o "tiny"
    output_key          = "sam2",       # Namespace en frame_data

    # --- Fuente de entrada ---
    input_source = "yolo:boxes",  # Modo bus: lee boxes de YOLOProcessor
    # input_source = [              # Modo manual: puntos iniciales
    #     {"points": [[cx, cy]], "labels": [1]},   # Slot 0
    #     {"points": [[cx, cy]], "labels": [1]},   # Slot 1
    # ],

    # --- Tracking de identidades ---
    max_entities         = 2,    # Número fijo de slots de identidad
    proximity_threshold  = 50,   # Distancia máxima entre frames (píxeles)
    area_tolerance       = 0.4,  # Variación de área permitida (±40%)
    iou_threshold        = 0.5,  # IoU para eliminar máscaras duplicadas
)
```

### Llamada a process()

```python
frame_data = sam.process(frame_data)

# Resultado disponible en (listas de longitud max_entities):
masks     = frame_data['sam2']['masks']      # list[np.ndarray] — binarias (H, W)
centroids = frame_data['sam2']['centroids']  # list[tuple]      — (x, y) o None
areas     = frame_data['sam2']['areas']      # list[float]      — píxeles² o 0
scores    = frame_data['sam2']['scores']     # list[float]      — 0.0–1.0 o 0.0
```

---

## Modos de entrada

### Modo Bus (`input_source="namespace:key"`)

Lee la salida de un procesador anterior y segmenta usando sus bounding boxes.

```python
# SAM2 lee frame_data['yolo']['boxes']
sam = SAM2Processor(input_source="yolo:boxes", max_entities=2)
```

Flujo interno:
```python
# (core/sam_processor.py:168)
def _segment_from_boxes(self, boxes):
    all_masks, all_scores = [], []
    for box in boxes[:self.max_entities]:
        masks, scores, _ = self.predictor.predict(
            box=np.array(box),
            multimask_output=False
        )
        if masks is not None and len(masks) > 0:
            all_masks.append(masks[0].astype(bool))
            all_scores.append(float(scores[0]))
    return all_masks, all_scores
```

### Modo Manual (`input_source=[{...}, ...]`)

Inicializa con puntos en el **primer frame**, luego usa los centroides de la máscara para los frames siguientes (tracking automático).

```python
sam = SAM2Processor(
    input_source=[
        {"points": [[320, 240]], "labels": [1]},  # Slot 0: punto en objeto A
        {"points": [[640, 360]], "labels": [1]},  # Slot 1: punto en objeto B
    ],
    max_entities=2
)
```

**Primer frame** — segmentación por puntos:
```python
# (core/sam_processor.py:118)
def _segment_from_manual_prompts(self):
    for prompt in self.input_source:
        pts  = np.array(prompt["points"])
        lbls = np.array(prompt.get("labels", [1] * len(pts)))
        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            multimask_output=False
        )
```

**Frames siguientes** — tracking por centroide:
```python
# (core/sam_processor.py:147)
def _segment_from_centroids(self):
    for slot_idx in active_slots:
        pt = self.matcher.prev_centroids[slot_idx]  # centroide del frame anterior
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([pt]),
            point_labels=np.array([1]),
            multimask_output=False
        )
```

---

## Sistema de tracking de identidades

### IdentityMatcher

Mantiene **slots fijos** de identidad para que el objeto A siempre sea el slot 0 y el objeto B siempre sea el slot 1, independientemente del orden de detección.

```python
# (tracking/identity_matcher.py:12)
class IdentityMatcher:
    def __init__(self, max_entities, proximity_threshold=50.0, area_tolerance=0.4):
        self.prev_centroids = [None] * max_entities  # centroide del frame anterior
        self.prev_areas     = [None] * max_entities  # área del frame anterior
        self.prev_masks     = [None] * max_entities  # máscara del frame anterior
```

**Algoritmo de matching** (por frame):

```
Para cada slot activo (con centroide previo conocido):
    1. Calcular distancia euclidiana a cada detección actual
    2. Ordenar detecciones por distancia ascendente
    3. Para la más cercana:
        a. dist < proximity_threshold    ✓
        b. |area_actual - area_prev| / area_prev < area_tolerance  ✓
    4. Si ambas condiciones se cumplen → asignar al slot
    5. Si no → el slot queda vacío (None) en este frame

Detecciones no asignadas → llenar slots vacíos en orden
```

```python
# (tracking/identity_matcher.py:112)
def _match_to_previous(self, curr_masks, curr_centroids, curr_areas, curr_scores):
    dist_matrix = cdist(curr_centroids, valid_prev_centroids)

    for slot_idx in valid_prev_indices:
        dists = dist_matrix[:, slot_position]
        for i_curr in np.argsort(dists):
            dist_ok = dists[i_curr] < self.proximity_threshold
            var_area = abs(curr_areas[i_curr] - prev_area) / prev_area
            area_ok  = var_area <= self.area_tolerance
            if dist_ok and area_ok:
                matched_masks[slot_idx] = curr_masks[i_curr]
                matched_centroids[slot_idx] = curr_centroids[i_curr]
                matched_areas[slot_idx] = curr_areas[i_curr]
                break
```

### Cálculo de centroide

```python
# (tracking/identity_matcher.py:180)
@staticmethod
def _get_centroid(mask: np.ndarray):
    y, x = np.where(mask)
    if len(x) == 0:
        return None
    return (np.mean(x), np.mean(y))  # (col_media, fila_media) → (x, y)
```

---

## Deduplicación de máscaras

Antes del matching, se eliminan máscaras redundantes por IoU:

```python
# (tracking/mask_utils.py:21)
def filter_duplicates(masks, scores, max_entities, iou_threshold=0.5):
    # Ordenar por área (mayor primero — más informativa)
    areas = [m.sum() for m in masks]
    sorted_idx = np.argsort(areas)[::-1]

    unique_masks = []
    for m_curr, score_curr in zip(ordered_masks, ordered_scores):
        is_duplicate = any(
            calculate_iou(m_curr, m_kept) > iou_threshold
            for m_kept in unique_masks
        )
        if not is_duplicate:
            unique_masks.append(m_curr)
            if len(unique_masks) >= max_entities:
                break

    return unique_masks, unique_scores
```

```python
# (tracking/mask_utils.py:5)
def calculate_iou(m1, m2):
    intersection = np.logical_and(m1, m2).sum()
    union        = np.logical_or(m1, m2).sum()
    return intersection / union if union > 0 else 0
```

---

## TrajectoryTracker

Historial de posiciones para dibujar trayectorias:

```python
# (tracking/trajectory_tracker.py)
class TrajectoryTracker:
    def __init__(self, max_length=30):
        self.history = {}  # {slot_idx: deque(maxlen=max_length)}

    def update(self, slot_idx, centroid):
        if slot_idx not in self.history:
            self.history[slot_idx] = deque(maxlen=self.max_length)
        self.history[slot_idx].append(centroid)

    def get_trajectory(self, slot_idx):
        return list(self.history.get(slot_idx, []))
```

---

## Flujo completo de SAM2Processor.process()

```python
# (core/sam_processor.py:210)
def process(self, frame_data):
    frame = frame_data['frame']

    # 1. Preparar imagen (obligatorio antes de predict)
    self.predictor.set_image(frame)

    # 2. Obtener máscaras según el modo
    if not self.first_frame_done and self.is_manual_mode:
        all_masks, all_scores = self._segment_from_manual_prompts()
    elif self.first_frame_done and self.is_manual_mode:
        all_masks, all_scores = self._segment_from_centroids()
    elif self.is_bus_mode:
        boxes = self._read_from_bus(frame_data)
        all_masks, all_scores = self._segment_from_boxes(boxes)

    # 3. Eliminar duplicados
    unique_masks, unique_scores = filter_duplicates(
        all_masks, all_scores, self.max_entities, self.iou_threshold
    )

    # 4. Asignar a slots de identidad
    matched_data = self.matcher.match(unique_masks, unique_scores)
    # matched_data = (masks[max_entities], centroids[max_entities],
    #                 areas[max_entities], scores[max_entities])

    # 5. Escribir en frame_data
    output_masks, output_centroids, output_areas, output_scores = \
        self._format_output(matched_data, frame.shape)

    frame_data[self.output_key] = {
        'masks':     output_masks,
        'centroids': output_centroids,
        'areas':     output_areas,
        'scores':    output_scores,
    }
    self.first_frame_done = True
    return frame_data
```

---

## Formato de salida detallado

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `masks` | `list[np.ndarray\|np.zeros]` | `max_entities` máscaras binarias `[H, W]`. Slot vacío = array de ceros |
| `centroids` | `list[tuple\|None]` | `max_entities` tuplas `(x, y)`. Slot vacío = `None` |
| `areas` | `list[float]` | `max_entities` floats. Slot vacío = `0` |
| `scores` | `list[float]` | `max_entities` floats 0.0–1.0. Slot vacío = `0.0` |

> Las listas siempre tienen exactamente `max_entities` elementos para garantizar indexación consistente por slot.

---

## Ejemplo: usar SAM2 sin pipeline

```python
import cv2
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tracking.identity_matcher import IdentityMatcher
from tracking.mask_utils import filter_duplicates

# Configuración
SAM2_DIR = Path("path/to/sam2")
model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_l.yaml",
    str(SAM2_DIR / "checkpoints/sam2.1_hiera_large.pt"),
    device="cuda"
)
predictor  = SAM2ImagePredictor(model)
matcher    = IdentityMatcher(max_entities=2, proximity_threshold=50)
MAX_ENTITIES = 2

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener boxes de YOLO (o cualquier detector)
    yolo_boxes = [...]  # lista de [x1, y1, x2, y2]

    # Segmentar con SAM2
    predictor.set_image(frame)
    masks, scores = [], []
    for box in yolo_boxes[:MAX_ENTITIES]:
        m, s, _ = predictor.predict(box=np.array(box), multimask_output=False)
        if m is not None and len(m) > 0:
            masks.append(m[0].astype(bool))
            scores.append(float(s[0]))

    # Deduplicar y matchear
    unique_masks, unique_scores = filter_duplicates(masks, scores, MAX_ENTITIES)
    matched = matcher.match(unique_masks, unique_scores)
    matched_masks, matched_centroids, matched_areas, matched_scores = matched

    # matched_masks[0] → slot 0 siempre es el mismo objeto
    # matched_masks[1] → slot 1 siempre es el mismo objeto
```
