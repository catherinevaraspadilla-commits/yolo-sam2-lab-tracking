# YOLOv8 + SAM2 Toolkit — Documentación Técnica

Esta carpeta contiene la documentación técnica del proyecto para facilitar la reutilización de sus componentes en otros programas.

## Documentos disponibles

| Archivo | Contenido |
|--------|-----------|
| [01_architecture.md](01_architecture.md) | Arquitectura general del pipeline y patrón de datos |
| [02_yolo_integration.md](02_yolo_integration.md) | Cómo se usa YOLOv8: carga, inferencia, filtros y salidas |
| [03_sam2_integration.md](03_sam2_integration.md) | Cómo se usa SAM2: carga, modos de entrada, salidas y tracking |
| [04_pipeline_guide.md](04_pipeline_guide.md) | Integración YOLO + SAM2 y cómo crear capas personalizadas |
| [05_quick_reference.md](05_quick_reference.md) | Cheatsheet de parámetros, formatos y ejemplos mínimos |

## Flujo general del sistema

```
Video/Imagen
    ↓
[YOLOProcessor]      → detecta objetos, devuelve boxes + clases
    ↓
[SAM2Processor]      → segmenta usando los boxes de YOLO
    ↓
[VisualizationProcessor] → dibuja máscaras, boxes, trayectorias
    ↓
Salida: MP4 / JSON / Frames
```

## Requisitos clave

- `ultralytics >= 8.0.0`
- `numpy < 2.0.0` (NumPy 1.x obligatorio)
- `torch` (con CUDA recomendado)
- `opencv-python >= 4.8.0`
- `scipy >= 1.10.0`
