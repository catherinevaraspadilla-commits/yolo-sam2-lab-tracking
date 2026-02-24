# Guía del Pipeline y Capas Personalizadas

**Archivo fuente:** [core/pipeline.py](../core/pipeline.py), [core/ExampleOfLayer.py](../core/ExampleOfLayer.py)

---

## ProcessMedia — Orquestador principal

### Constructor

```python
from core.pipeline import ProcessMedia

pipeline = ProcessMedia(
    source      = "data/input/video.mp4",  # Ruta a video o imagen
    processors  = [...],                   # Lista ordenada de procesadores
    output      = ["video", "json", "frames"],  # Tipos de salida
    output_dir  = "data/output",           # Directorio de salida
    use_vis_frame = True,                  # Usar vis_frame si existe
)
pipeline.run()
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `source` | `str` | Ruta al video (`.mp4`, `.avi`, etc.) o imagen (`.jpg`, `.png`) |
| `processors` | `list` | Procesadores en orden de ejecución |
| `output` | `str\|list` | `"video"`, `"json"`, `"frames"` o combinación |
| `output_dir` | `str` | Directorio raíz de salida |
| `use_vis_frame` | `bool` | Si `True`, guarda `vis_frame`; si `False`, guarda el frame original |

### Carpeta de salida

Cada ejecución crea una carpeta con timestamp:
```
data/output/
  └── nombre_video_20250224_153045/
        ├── nombre_video.mp4       # output="video"
        ├── output.json            # output="json"
        └── frames/                # output="frames"
              ├── frame_00000.jpg
              ├── frame_00001.jpg
              └── ...
```

---

## Configuraciones completas de ejemplo

### Ejemplo 1: Solo YOLO (detección básica)

```python
from core.pipeline import ProcessMedia
from core.yolo_processor import YOLOProcessor
from core.visualization import VisualizationProcessor

pipeline = ProcessMedia(
    source="data/input/video.mp4",
    processors=[
        YOLOProcessor(
            model="yolov8n.pt",
            confidence=0.5,
            max_entities=5
        ),
        VisualizationProcessor(
            input_keys={"yolo": ["boxes", "confidences"]},
            show_boxes=True,
        )
    ],
    output=["video", "json"]
)
pipeline.run()
```

### Ejemplo 2: YOLO + SAM2 (pipeline completo)

```python
from core.pipeline import ProcessMedia
from core.yolo_processor import YOLOProcessor
from core.sam_processor import SAM2Processor
from core.visualization import VisualizationProcessor

pipeline = ProcessMedia(
    source="data/input/video.mp4",
    processors=[
        YOLOProcessor(
            model="models/yolo_8l_rat.pt",
            confidence=0.6,
            entities=2
        ),
        SAM2Processor(
            input_source="yolo:boxes",  # Consume la salida de YOLO
            model_type="large",
            max_entities=2
        ),
        VisualizationProcessor(
            input_keys={
                "yolo": ["boxes", "confidences"],
                "sam2": ["masks", "centroids", "scores"]
            },
            show_masks=True,
            show_boxes=True,
            show_trajectories=True,
            show_centroids=True,
            trail_length=30,
            mask_alpha=0.35
        )
    ],
    output=["video", "json", "frames"]
)
pipeline.run()
```

### Ejemplo 3: SAM2 con puntos manuales (sin YOLO)

```python
pipeline = ProcessMedia(
    source="data/input/video.mp4",
    processors=[
        SAM2Processor(
            input_source=[
                {"points": [[320, 240]], "labels": [1]},
                {"points": [[640, 480]], "labels": [1]},
            ],
            model_type="large",
            max_entities=2,
            proximity_threshold=80,   # Más tolerante al movimiento
            area_tolerance=0.5
        ),
        VisualizationProcessor(
            input_keys={"sam2": ["masks", "centroids"]},
            show_masks=True,
            show_centroids=True,
            show_trajectories=True
        )
    ],
    output="video"
)
pipeline.run()
```

---

## Crear una capa personalizada

Cualquier clase que implemente `process(frame_data) -> dict` puede insertarse en el pipeline.

### Plantilla mínima

```python
class MiProcesador:
    """Descripción del procesador"""

    def __init__(self, mi_param="valor"):
        self.mi_param = mi_param
        self.output_key = "mi_namespace"  # Clave en frame_data

    def process(self, frame_data: dict) -> dict:
        # 1. Leer datos necesarios
        frame = frame_data['frame']

        # 2. Procesar
        mi_resultado = self._hacer_algo(frame)

        # 3. Escribir en frame_data (sin borrar claves existentes)
        frame_data[self.output_key] = {
            'resultado': mi_resultado
        }

        # 4. SIEMPRE devolver frame_data
        return frame_data

    def _hacer_algo(self, frame):
        # Lógica interna
        return None
```

### Capa que consume YOLO

```python
class ContadorDeObjetos:
    def __init__(self, yolo_key="yolo"):
        self.yolo_key = yolo_key

    def process(self, frame_data: dict) -> dict:
        yolo_data = frame_data.get(self.yolo_key, {})
        boxes = yolo_data.get('boxes', [])
        count = len(boxes)

        # Añadir al metadata (aparece en el JSON de salida)
        frame_data['metadata']['object_count'] = count

        # Dibujar en el frame directamente (si no hay VisualizationProcessor)
        frame = frame_data['frame'].copy()
        import cv2
        cv2.putText(frame, f"Objetos: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_data['vis_frame'] = frame

        return frame_data
```

### Capa que consume SAM2

```python
import numpy as np

class AnalizadorDeMascaras:
    def __init__(self, sam_key="sam2"):
        self.sam_key = sam_key

    def process(self, frame_data: dict) -> dict:
        sam_data = frame_data.get(self.sam_key, {})
        masks     = sam_data.get('masks', [])
        centroids = sam_data.get('centroids', [])
        areas     = sam_data.get('areas', [])

        analisis = []
        for slot_idx, (mask, centroid, area) in enumerate(zip(masks, centroids, areas)):
            if mask is None or centroid is None:
                continue  # Slot inactivo

            # Calcular bounding box de la máscara
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            analisis.append({
                'slot':     slot_idx,
                'centroid': centroid,
                'area':     area,
                'bbox':     [cmin, rmin, cmax, rmax]  # [x1, y1, x2, y2]
            })

        frame_data['analisis_mascaras'] = analisis
        frame_data['metadata']['mascaras_activas'] = len(analisis)
        return frame_data
```

### Capa con validación de dependencias

```python
class MiProcesadorConValidacion:
    def validate(self, previous_processors):
        """
        Se llama antes de procesar el primer frame.
        Verifica que las capas necesarias estén en el pipeline.
        """
        has_yolo = any(
            hasattr(p, 'output_key') and p.output_key == 'yolo'
            for p in previous_processors
        )
        if not has_yolo:
            raise ValueError("MiProcesador requiere YOLOProcessor antes en el pipeline")

    def process(self, frame_data: dict) -> dict:
        ...
        return frame_data
```

---

## Acceso al frame anterior

```python
def process(self, frame_data: dict) -> dict:
    prev = frame_data.get('previous_data')
    if prev is not None:
        prev_masks = prev.get('sam2', {}).get('masks', [])
        # Comparar con máscaras del frame anterior
    ...
    return frame_data
```

---

## VisualizationProcessor — Referencia rápida

```python
from core.visualization import VisualizationProcessor

vis = VisualizationProcessor(
    # Qué datos leer del bus
    input_keys = {
        "yolo": ["boxes", "confidences", "labels", "keypoints"],
        "sam2": ["masks", "centroids", "scores"]
    },

    # Qué dibujar
    show_masks        = True,   # Overlay de máscara con color por slot
    show_boxes        = True,   # Rectángulo del bounding box
    show_trajectories = True,   # Línea de trayectoria histórica
    show_keypoints    = False,  # Keypoints del modelo pose
    show_centroids    = True,   # Punto en el centroide

    # Estilo
    trail_length           = 30,   # Frames de historial de trayectoria
    box_thickness          = 2,    # Grosor del rectángulo
    font_scale             = 0.6,  # Tamaño del texto
    mask_alpha             = 0.35, # Transparencia de la máscara (0=invisible, 1=opaco)
    mask_border_thickness  = 2,    # Grosor del borde de la máscara
    trail_thickness        = 2,    # Grosor de la línea de trayectoria
)
```

El procesador escribe el frame renderizado en `frame_data['vis_frame']`, que es el que usa `ProcessMedia` para guardar el video.

---

## Orden correcto de procesadores

```
[Detectores]              → generan boxes/clases/keypoints
    ↓
[Segmentadores]           → consumen boxes, generan máscaras
    ↓
[Analizadores/Contadores] → consumen cualquier dato previo
    ↓
[VisualizationProcessor]  → siempre al final, dibuja sobre el frame
```

El orden importa porque cada procesador solo puede leer datos de los procesadores que vinieron antes.
