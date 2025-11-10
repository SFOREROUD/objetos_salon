# Inventario Automático del Salón de Cómputo

**Estudiante:** Sebastian Forero
**Código:** 20242595001
**Maestría:** Ciencias de la Computación y las Comunicaciones
**Curso:** BIG DATA - Módulo de Redes Convolucionales
**Profesor:** Gerardo Muñoz
**Fecha:** Noviembre 2024

---

## 📋 Descripción del Proyecto

Sistema web de detección y conteo de objetos del salón de cómputo usando Deep Learning. El sistema identifica y cuenta automáticamente 6 tipos de objetos mediante un modelo TensorFlow Lite que se ejecuta completamente de forma local en el navegador.

### Objetos Detectables

| Código | Objeto   |
|--------|----------|
| 0      | CPU      |
| 1      | Mesa     |
| 2      | Mouse    |
| 3      | Pantalla |
| 4      | Silla    |
| 5      | Teclado  |

---

## 🧠 Modelo de Deep Learning

### Arquitectura
- **Tipo:** SSD (Single Shot Detector) con MobileNetV2 backbone
- **Input:** 320×320×3 (RGB)
- **Framework:** TensorFlow/Keras → TensorFlow Lite
- **Tamaño del modelo:** 16 MB
- **Formato:** TFLite optimizado para web

### Outputs del Modelo

El modelo genera 3 salidas:

1. **BBox** `[1, 4]`: Coordenadas normalizadas del bounding box `[ymin, xmin, ymax, xmax]`
2. **Class** `[1, 6]`: Probabilidades de cada clase (CPU, Mesa, Mouse, Pantalla, Silla, Teclado)
3. **Confidence** `[1, 1]`: Score de confianza de la detección (0-1)

---

## 📊 Entrenamiento del Modelo

### Dataset

Se generó un dataset anotado automáticamente usando **pseudo-labeling**:

1. **Extracción de frames:** 228 imágenes extraídas de 7 videos del salón
2. **Etiquetado automático:** Usando SSD MobileNet V2 COCO preentrenado
3. **Resultado:** 190 imágenes con 752 detecciones (bounding boxes)
4. **Split:** 80% entrenamiento / 20% validación

### Proceso de Entrenamiento

#### Fase 1: Pseudo-labeling
```bash
python generar_etiquetas_auto.py
```
- Descarga SSD MobileNet V2 COCO preentrenado
- Detecta objetos relevantes en las 228 imágenes
- Genera anotaciones en formato YOLO (txt)
- Mapeo de clases COCO a nuestras clases:
  - COCO 62 (chair) → Silla
  - COCO 67 (dining table) → Mesa
  - COCO 72 (tv) → Pantalla
  - COCO 73 (laptop) → CPU
  - COCO 74 (mouse) → Mouse
  - COCO 76 (keyboard) → Teclado

#### Fase 2: Entrenamiento del Detector
```bash
python entrenar_ssd_tflite.py
```

**Transfer Learning en 2 fases:**

1. **Fase 1 (30 épocas):** Entrenar solo la cabeza de detección
   - Backbone MobileNetV2 congelado (pesos de ImageNet)
   - Learning rate: 0.001

2. **Fase 2 (70 épocas):** Fine-tuning completo
   - Descongelar backbone completo
   - Learning rate: 0.0001

**Técnicas utilizadas:**
- Data augmentation: flip horizontal
- Early stopping (patience=15)
- ReduceLROnPlateau (factor=0.5, patience=5)
- Loss multi-objetivo:
  - Classification loss (sparse categorical crossentropy)
  - BBox regression loss (MSE)
  - Confidence loss (binary crossentropy)

#### Fase 3: Conversión a TFLite
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

---

## 🚀 Uso de la Aplicación

### Requisitos
- Navegador web moderno (Chrome, Firefox, Edge)
- Servidor web local (no funciona con `file://`)

### Instrucciones

1. **Iniciar servidor local:**
   ```bash
   cd inventario
   python -m http.server 8000
   ```

2. **Abrir en navegador:**
   ```
   http://localhost:8000
   ```

3. **Usar la aplicación:**
   - Hacer clic en "Seleccionar Imagen" o arrastrar una imagen
   - Hacer clic en "🔍 Detectar Objetos"
   - Ver resultados:
     - Imagen con bounding boxes azules
     - Número de clase dentro de cada box
     - Tabla de conteo por tipo de objeto

---

## 🏗️ Arquitectura Técnica

### Frontend
- **HTML5** con diseño responsive
- **CSS3** con gradientes y animaciones
- **JavaScript ES6+** con async/await
- **TensorFlow.js TFLite** para inferencia

### Algoritmo de Detección

La aplicación usa **sliding window** con Non-Maximum Suppression (NMS):

1. **Sliding Window:** Recorre la imagen con ventanas de múltiples tamaños
2. **Predicción:** Cada ventana se procesa con el modelo TFLite
3. **Filtrado:** Se descartan detecciones con confidence < 0.25
4. **NMS:** Se eliminan detecciones duplicadas (IoU > 0.45)
5. **Visualización:** Dibuja bounding boxes azules con número de clase

### Optimizaciones

- **Gestión de memoria:** `tf.tidy()` para liberar tensores
- **Procesamiento adaptativo:**
  - Imágenes pequeñas (<320px): procesamiento completo
  - Imágenes grandes: sliding window con stride=80
- **Cache de modelo:** Se carga una sola vez al inicio

---

## 📁 Estructura de Archivos

```
inventario/
├── index.html              # Aplicación web principal
├── js/
│   └── detector_salon.js   # Lógica de detección y NMS
├── models/
│   ├── detector_salon.tflite  # Modelo TFLite (16MB)
│   └── model_info.json         # Metadatos del modelo
└── README.md               # Esta documentación
```

---

## 🔧 Dependencias

### Para Entrenamiento (Python)
```
tensorflow>=2.13.0
opencv-python
numpy
scikit-learn
```

### Para Inferencia (Web)
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/tf-tflite.min.js"></script>
```

---

## 📈 Resultados

### Métricas del Dataset
- **Tasa de detección:** 83.3% (190/228 imágenes)
- **Detecciones totales:** 752 bounding boxes
- **Promedio por imagen:** ~4 objetos

### Performance de Inferencia
- **Tiempo de carga del modelo:** ~2 segundos
- **Tiempo de detección:**
  - Imagen 128×128: ~500ms
  - Imagen 640×480: ~2-3 segundos
  - Imagen 1280×720: ~5-8 segundos

### Características del Modelo
- **Tamaño:** 16 MB (TFLite optimizado)
- **Precisión:** Balanceada para 6 clases
- **Ejecución:** 100% local en navegador

---

## 🎯 Decisiones de Diseño

1. **Pseudo-labeling:** Permitió generar un dataset anotado sin etiquetado manual
2. **SSD MobileNetV2:** Balance óptimo entre precisión y tamaño
3. **Sliding Window:** Permite detectar múltiples objetos sin modificar arquitectura
4. **TFLite:** Optimización automática para reducir tamaño
5. **Threshold adaptativo:** 0.25 para mayor sensibilidad en detección

---

## 📝 Notas

- El modelo detecta múltiples objetos usando sliding window
- Los bounding boxes se dibujan en **color azul** con el **número de clase**
- El conteo es preciso gracias a Non-Maximum Suppression
- La aplicación funciona completamente offline

---

## 👤 Autor

**Sebastian Forero**
Código: 20242595001
Maestría en Ciencias de la Computación y las Comunicaciones
Curso: BIG DATA - Módulo de Redes Convolucionales
Profesor: Gerardo Muñoz
