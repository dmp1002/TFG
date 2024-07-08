# VitiScan: Procesamiento de Imágenes Hiperespectrales para el Análisis de Hojas de Vid

**VitiScan** es una aplicación diseñada para procesar imágenes hiperespectrales de hojas de vid. Estas hojas han sido tratadas con un producto antifúngico que contiene cobre y azufre en diferentes concentraciones. El objetivo de VitiScan es etiquetar cada píxel de la imagen como fondo, hoja de vid o gota con producto antifúngico. A esta clasificación se le denomina *trinarización*, donde los valores asignados son:

- Fondo: 00
- Hoja de vid: 01
- Gota con producto antifúngico: 10

El proceso de trinarización implica dos etapas de binarización en diferentes bandas espectrales extraídas de la imagen, seguidas de transformaciones morfológicas. Una vez completado, se aplica esta trinarización a cada hoja procesada.

Los datos resultantes de la trinarización se presentan en tres formatos:

1. **Imagen Trinarizada**: Representación visual a tres colores que muestra los valores de la trinarización.
2. **Archivo CSV (píxeles)**: Contiene los valores trinarizados de cada píxel.
3. **Archivo CSV (porcentajes)**: Incluye el porcentaje de área de hoja y el porcentaje de área cubierta por las gotas en la hoja.

## Funcionalidades de la Aplicación

VitiScan dispone de tres pestañas con funciones diferentes:

1. **Trinarizar**: Permite la trinarización de una imagen hiperespectral individual.
2. **Trinarizar por lotes**: Permite procesar varias imágenes hiperespectrales al mismo tiempo.
3. **Visualizar**: Permite crear composiciones superpuestas para analizar los resultados.

**Autor**: David Merinero Porres
