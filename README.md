# Logo Detection & Classification
Proyecto desarrollado para la asignatura de Aprendizaje Automático (UDC). Sistema capaz de detectar falsificaciones de logos de marca utilizando técnicas de visión artificial.

## Tecnologías
- **Lenguaje:** Julia
- **Deep Learning:** Flux.jl (Redes Neuronales Convolucionales - CNN)
- **Modelos Clásicos:** SVM, k-NN, Redes Neuronales Artificiales (RNA)

## Funcionalidades
- **Preprocesado:** Normalización de imágenes y extracción de características (pipelines automatizados).
- **Clasificación:** Comparativa de rendimiento entre modelos clásicos y arquitecturas de Deep Learning.
- **Evaluación:** Análisis mediante métricas de rendimiento (F1-Score, Recall, Matriz de Confusión).

## Estructura del repositorio
- `aproxAA.jl`: Implementación de modelos clásicos (k-NN, SVM, RNA).
- `aproxDeepLearning.jl`: Implementación de la red neuronal convolucional (CNN) con Flux.
- `DatasetLogos/`: Carpeta que contiene las imágenes de entrenamiento (Adidas Real/Fake).
