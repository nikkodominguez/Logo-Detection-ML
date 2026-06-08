# Logo Detection & Classification
Proyecto desarrollado para la asignatura de Aprendizaje Automático (UDC). Sistema capaz de detectar falsificaciones de logos de cierta marca utilizando técnicas de visión artificial.

## Tecnologías
- **Lenguaje:** Julia
- **Deep Learning:** Flux.jl (Redes Neuronales Convolucionales - CNN)
- **Modelos Clásicos:** Redes Neuronales Artificiales (RNA), SVM, Árboles de decisión, k-NN, DoME.
## Funcionalidades
- **Preprocesado:** Normalización de imágenes y extracción de características (pipelines automatizados).
- **Clasificación:** Comparativa de rendimiento entre modelos clásicos y arquitecturas de Deep Learning.
- **Evaluación:** Análisis mediante métricas de rendimiento (F1-Score, Recall, Matriz de Confusión).

## Estructura del repositorio
- `DatasetLogos/`: Carpeta que contiene las imágenes de entrenamiento (Adidas Real/Fake).
- `fonts/`: Carpeta que contiene los módulos de código fuente del proyecto:
  - `firmas.jl`: Librería base de Machine Learning. Contiene las implementaciones manuales de métricas (Matriz de confusión, F1, Recall), validación cruzada, normalización de datos (*One-Hot Encoding*, *MinMax*, *Zero-Mean*) y los *wrappers* de entrenamiento de todos los modelos predictivos.
  - `procesamientoImagenes.jl`: *Pipeline* de extracción de características para los modelos clásicos. Implementa umbrales dinámicos, segmentación en cuadrículas adaptativas 3x3 y cálculo de variables globales (relación de aspecto y densidad de píxeles).
  - `visualizarZonas.jl`: Script de depuración de visión artificial. Utiliza perfiles de proyección y operaciones morfológicas para localizar los *bounding boxes* exactos y verificar empíricamente la separación espacial entre el isotipo y la tipografía del logo.
  - `pruebas0.jl` / `pruebas1.jl`: Código de prueba para comprobar el correcto funcionamiento de las funciones de firmas.jl
- `aproxAA.jl`: Implementación de modelos clásicos (RNA, SVM, Árboles de decisión, k-NN, DoMe).
- `aproxDeepLearning.jl`: Implementación de la red neuronal convolucional (CNN) con Flux.
