# Integración de YOLOv8

**Archivo fuente:** [core/yolo_processor.py](../core/yolo_processor.py)

---

## Carga del modelo

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")          # Modelo preentrenado COCO
model = YOLO("models/mi_modelo.pt") # Modelo personalizado
```

Los modelos disponibles van de menor a mayor (velocidad vs precisión):
- `yolov8n.pt` — nano
- `yolov8s.pt` — small
- `yolov8m.pt` — medium
- `yolov8l.pt` — large
- `yolov8x.pt` — extra large

---

## Inferencia directa (sin pipeline)

```python
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
frame = cv2.imread("imagen.jpg")

results = model(frame, conf=0.5, verbose=False)[0]

# Extraer resultados
boxes       = results.boxes.xyxy.cpu().numpy()   # [N, 4] — x1, y1, x2, y2
classes     = results.boxes.cls.cpu().numpy()    # [N]    — índice de clase
confidences = results.boxes.conf.cpu().numpy()   # [N]    — score 0.0–1.0
names       = model.names                        # dict {id: "nombre"}

# Máscaras (solo si el modelo las produce, e.g. yolov8n-seg.pt)
masks = [m.cpu().numpy() for m in results.masks.data] if results.masks else []

# Keypoints (solo modelos pose, e.g. yolov8n-pose.pt)
keypoints = [k.cpu().numpy() for k in results.keypoints.data] if results.keypoints else []
```

---

## Clase YOLOProcessor

### Constructor

```python
from core.yolo_processor import YOLOProcessor

yolo = YOLOProcessor(
    model        = "yolov8n.pt",   # Ruta al modelo
    confidence   = 0.5,            # Umbral de confianza
    output_key   = "yolo",         # Namespace en frame_data

    # --- Filtros de cantidad ---
    entities     = None,  # Mantener exactamente N detecciones
    max_entities = None,  # Mantener máximo N detecciones
    min_entities = None,  # Si hay menos de N, baja el umbral y reintenta

    # --- Filtros de área ---
    area_min     = None,  # Área mínima del bounding box (píxeles²)
    area_max     = None,  # Área máxima del bounding box (píxeles²)
    area         = None,  # Área adaptativa (aprende del primer frame)
    area_error   = 0.2,   # Tolerancia del área adaptativa (±20%)

    # --- Filtros de clase ---
    classes         = None,  # Lista de IDs a incluir, e.g. [0, 2]
    exclude_classes = None,  # Lista de IDs a excluir, e.g. [1]

    # --- Filtros espaciales ---
    max_overlap  = None,  # Umbral IoU para NMS personalizado
    edge_margin  = 0,     # Margen de borde en píxeles a ignorar
    roi          = None,  # Región de interés [x1, y1, x2, y2]
)
```

### Llamada a process()

```python
frame_data = {'frame': frame, 'frame_index': 0, 'metadata': {}}
frame_data = yolo.process(frame_data)

# Resultado disponible en:
boxes       = frame_data['yolo']['boxes']        # np.ndarray [N, 4]
classes     = frame_data['yolo']['classes']      # np.ndarray [N]
labels      = frame_data['yolo']['labels']       # list[str]
confidences = frame_data['yolo']['confidences']  # np.ndarray [N]
masks       = frame_data['yolo']['masks']        # list (vacío si modelo base)
keypoints   = frame_data['yolo']['keypoints']    # list (vacío si modelo base)
```

---

## Lógica de filtrado (en orden de aplicación)

### 1. Umbral de confianza
```python
results = self.model(frame, conf=self.confidence, verbose=False)[0]
# YOLO descarta internamente detecciones por debajo del umbral
```

### 2. Filtros de clase
```python
# Solo incluir clases específicas (core/yolo_processor.py:235)
if self.classes is not None:
    mask = np.isin(classes, self.classes)
    boxes, classes, confidences = boxes[mask], classes[mask], confidences[mask]

# Excluir clases específicas (core/yolo_processor.py:239)
if self.exclude_classes is not None:
    mask = ~np.isin(classes, self.exclude_classes)
    boxes, classes, confidences = boxes[mask], classes[mask], confidences[mask]
```

### 3. Filtro de área
```python
# Calcular área de cada box
areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# Filtro fijo
if self.area_min: mask &= areas >= self.area_min
if self.area_max: mask &= areas <= self.area_max

# Filtro adaptativo: aprende del primer frame
if self.area is not None:
    ref_area = self.area  # Se fija en el primer frame con detección
    low  = ref_area * (1 - self.area_error)
    high = ref_area * (1 + self.area_error)
    mask &= (areas >= low) & (areas <= high)
```

### 4. Filtro de borde
```python
# Elimina detecciones cuyo centroide está dentro del margen de borde
# (core/yolo_processor.py:180)
def _is_near_edge(self, box, frame_shape):
    h, w = frame_shape[:2]
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return (cx < self.edge_margin or cx > w - self.edge_margin or
            cy < self.edge_margin or cy > h - self.edge_margin)
```

### 5. Filtro de ROI
```python
# Mantiene detecciones cuyo centroide está dentro del ROI
# (core/yolo_processor.py:193)
def _is_in_roi(self, box, roi):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2
```

### 6. NMS personalizado
```python
# (core/yolo_processor.py:285) Si max_overlap está definido
# Ordena por confianza descendente
# Elimina boxes con IoU > max_overlap respecto a boxes ya aceptados
```

### 7. Control de cantidad
```python
# entities: mantener exactamente N (ordena por confianza)
# max_entities: mantener máximo N
# min_entities: si hay menos de N, baja confidence 5% y reintenta (recursivo)
```

---

## Función IoU interna

```python
# (core/yolo_processor.py:157)
def _calculate_iou(self, box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0
```

---

## Ejemplos de uso

### Detección básica con filtro de clase
```python
yolo = YOLOProcessor(
    model="yolov8n.pt",
    confidence=0.5,
    classes=[0],          # Solo personas (clase 0 en COCO)
    max_entities=5
)
```

### Detección en zona específica
```python
yolo = YOLOProcessor(
    model="models/detector.pt",
    confidence=0.6,
    roi=[100, 100, 800, 600],  # Solo dentro de esta zona
    edge_margin=20,             # Ignorar borde de 20px
    entities=2                  # Exactamente 2 detecciones
)
```

### Detección con área adaptativa
```python
# Útil cuando el objeto siempre debe tener un tamaño similar
yolo = YOLOProcessor(
    model="models/detector.pt",
    confidence=0.5,
    area=None,       # Se aprende automáticamente del primer frame
    area_error=0.3   # ±30% de variación permitida
)
```

### Exportar nombres de clases del modelo
```python
yolo = YOLOProcessor(model="yolov8n.pt")
yolo._save_model_classes("clases_yolo.json")
# Genera: {"0": "person", "1": "bicycle", "2": "car", ...}
```

---

## Formato de salida detallado

| Campo | Tipo | Shape / Contenido |
|-------|------|-------------------|
| `boxes` | `np.ndarray` | `[N, 4]` — `[x1, y1, x2, y2]` en píxeles |
| `classes` | `np.ndarray` | `[N]` — índices enteros de clase |
| `labels` | `list[str]` | `[N]` — nombres de clase |
| `confidences` | `np.ndarray` | `[N]` — floats 0.0–1.0 |
| `masks` | `list` | `[N]` — arrays de máscara o lista vacía |
| `keypoints` | `list` | `[N]` — arrays de keypoints o lista vacía |

> Las boxes siempre están en coordenadas absolutas de píxel del frame original.
