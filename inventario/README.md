# 🎯 Inventario Automático del Salón de Cómputo

**Estudiante:** Sebastian Forero
**Código:** 20242595001
**Maestría en Ciencias de la Computación y las Comunicaciones**
**Curso:** BIG DATA - Módulo de Redes Convolucionales
**Profesor:** Gerardo Muñoz
**Fecha de entrega:** 9 de Noviembre 2024

---

## 🎬 Demostración del Sistema

![Demostración del Sistema de Detección](GIF_BIGDATA.gif)

---

## 📋 Descripción del Proyecto

Sistema web de detección y conteo automático de objetos del salón de cómputo usando **Deep Learning**. Implementa un modelo **SSD MobileNetV2** entrenado con **pseudo-labeling** que se ejecuta completamente de forma local en el navegador usando **TensorFlow Lite**.

### Objetos Detectables

| ID | Objeto   |
|----|----------|
| 0  | CPU      |
| 1  | Mesa     |
| 2  | Mouse    |
| 3  | Pantalla |
| 4  | Silla    |
| 5  | Teclado  |

**Características principales:**
- ✅ Detección de múltiples objetos simultáneos
- ✅ Bounding boxes en color azul con números de clase
- ✅ Conteo automático por categoría
- ✅ Ejecución 100% local (sin servidor backend)
- ✅ Interfaz web responsive y profesional

---

## 🏗️ Arquitectura Técnica

### Modelo: SSD MobileNetV2 + TFLite

- **Arquitectura:** SSD (Single Shot Detector)
- **Backbone:** MobileNetV2 (alpha=1.0) pre-entrenado en ImageNet
- **Entrada:** 320×320×3 píxeles
- **Salidas:**
  - BBox [1, 4]: Coordenadas normalizadas [ymin, xmin, ymax, xmax]
  - Class [1, 6]: Probabilidades de cada clase
  - Confidence [1, 1]: Score de confianza
- **Tamaño:** 16 MB (optimizado con Float16)
- **Formato:** TensorFlow Lite (.tflite)

### Método de Detección: Sliding Window + NMS

1. **Sliding Window:** Ventanas de 320, 384, 448 píxeles con stride de 80px
2. **Predicción:** Cada ventana procesada por el modelo SSD
3. **Filtrado:** Confidence threshold = 0.25
4. **NMS:** Elimina detecciones duplicadas (IoU > 0.45)
5. **Optimización:** Imágenes <320px se procesan completas (sin sliding)

### Stack Tecnológico

- **TensorFlow.js TFLite** - Inferencia del modelo
- **TensorFlow.js Core** - Operaciones de tensores
- **TensorFlow.js CPU Backend** - Backend de cómputo
- **HTML5 + CSS3** - Interfaz responsive
- **Canvas API** - Visualización de bounding boxes

---

## 📁 Estructura del Proyecto

```
Photos-1-001/
├── inventario/                        # ⭐ APLICACIÓN WEB (ENTREGA)
│   ├── index.html                     # Interfaz principal
│   ├── js/
│   │   └── detector_salon.js          # Lógica de detección
│   ├── models/
│   │   ├── detector_salon.tflite      # Modelo TFLite (16 MB)
│   │   └── model_info.json            # Metadatos del modelo
│   └── README.md                      # Documentación completa
│
├── generar_etiquetas_auto.py          # Script de pseudo-labeling
├── entrenar_ssd_tflite.py             # Script de entrenamiento
├── requirements.txt                   # Dependencias Python
│
├── VID_*.mp4                          # Videos originales (7 archivos)
└── README.md                          # Este archivo
```

---

## 🚀 Instalación y Uso

### Prerrequisitos

- Navegador moderno (Chrome, Firefox, Edge)
- Python 3.8+ (solo para servidor web local)

### Paso 1: Iniciar Servidor Web

```bash
cd inventario
python -m http.server 8000
```

### Paso 2: Abrir Aplicación

Abre en tu navegador: **http://localhost:8000**

### Paso 3: Detectar Objetos

1. Espera el mensaje: "✅ Modelo Detector del Salón cargado correctamente"
2. Sube una imagen del salón (JPG/PNG) arrastrándola o haciendo clic
3. Haz clic en "🔍 Detectar Objetos"
4. Espera el análisis (tiempo varía según tamaño de imagen)
5. Revisa los resultados:
   - Imagen con bounding boxes azules
   - Números de clase (0-5) dentro de cada box
   - Tabla de conteo por categoría

---

## 🎯 Características del Sistema

### ✅ Detección y Conteo
- Modelo SSD entrenado específicamente para el salón
- Sliding window adaptativo (multi-escala)
- Filtrado por confianza (threshold=0.25)
- Non-Maximum Suppression (NMS) con IoU=0.45
- Conteo automático de 6 categorías

### ✅ Interfaz Web
- Diseño moderno con gradientes purple/blue
- Drag & Drop para subir imágenes
- Visualización lado a lado (original vs detectado)
- Bounding boxes azules con número de clase
- Tabla de conteo con badges de color

### ✅ Optimización
- Modelo TFLite optimizado: 16 MB
- Float16 quantization
- Inferencia 100% local en navegador
- Gestión eficiente de memoria (`tf.tidy()`)
- Procesamiento adaptativo según tamaño de imagen

---

## 📊 Parámetros Configurables

En `inventario/js/detector_salon.js`:

```javascript
const INPUT_SIZE = 320;                  // Tamaño de entrada del modelo
const CONFIDENCE_THRESHOLD = 0.25;       // Umbral de confianza (0-1)
const NMS_THRESHOLD = 0.45;              // Umbral NMS para IoU (0-1)
const WINDOW_SIZES = [320, 384, 448];    // Tamaños de ventana (px)
const STRIDE = 80;                       // Paso de la ventana (px)
```

**Ajustes recomendados:**

- **Más detecciones:** `CONFIDENCE_THRESHOLD = 0.20`
- **Menos duplicados:** `NMS_THRESHOLD = 0.50`
- **Más rápido:** `STRIDE = 120` o usar solo `[320]`

---

## 📈 Rendimiento

### Métricas del Dataset
- **228 imágenes** extraídas de 7 videos
- **190 imágenes** con detecciones (83.3%)
- **752 bounding boxes** generados
- **~4 objetos** por imagen promedio

### Tiempos de Detección Estimados

| Resolución Imagen | Procesamiento | Tiempo Estimado |
|------------------|---------------|-----------------|
| 128×128          | Imagen completa | ~500ms       |
| 640×480          | Sliding window | 2-3s          |
| 1280×720         | Sliding window | 5-8s          |

---

## 🔧 Solución de Problemas

### Error: "Error cargando modelo"

**Causa:** No se está ejecutando desde servidor web

**Solución:**
```bash
cd inventario
python -m http.server 8000
```
No abras el archivo directamente (file://)

### Detecciones Inexactas

**Soluciones:**
1. Ajusta `CONFIDENCE_THRESHOLD` a 0.3
2. Entrena con más datos
3. Ajusta los tamaños de ventana

### Detección Lenta

**Soluciones:**
1. Aumenta `STRIDE` a 48 o 64
2. Usa solo 1-2 tamaños de ventana
3. Redimensiona la imagen antes de subirla

---

## 📝 Entrenamiento del Modelo

### Proceso Completo

#### 1. Generación del Dataset (Pseudo-labeling)

```bash
python generar_etiquetas_auto.py
```

- Descarga SSD MobileNet V2 COCO preentrenado
- Detecta objetos en las 228 imágenes extraídas
- Genera anotaciones YOLO automáticamente
- Resultado: 190 imágenes con 752 bounding boxes

#### 2. Entrenamiento del Detector

```bash
python entrenar_ssd_tflite.py
```

**Transfer Learning en 2 fases:**

**Fase 1 (30 épocas):** Entrenar detection head
- Backbone MobileNetV2 congelado
- Learning rate: 0.001
- Solo entrenan las capas de detección

**Fase 2 (70 épocas):** Fine-tuning completo
- Backbone descongelado
- Learning rate: 0.0001
- Todo el modelo se ajusta

**Técnicas utilizadas:**
- Transfer learning desde ImageNet
- Data augmentation (flip horizontal)
- Early stopping (patience=15)
- ReduceLROnPlateau (factor=0.5)
- Multi-task loss

#### 3. Exportación a TFLite

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

### Re-entrenar con tus datos

1. Coloca videos en la carpeta raíz
2. Ejecuta `python generar_etiquetas_auto.py`
3. Ejecuta `python entrenar_ssd_tflite.py`
4. El modelo se genera en `modelo_entrenado_ssd/`

---

## 🎯 Cumplimiento de Requisitos

| Criterio | Cumplimiento | Detalles |
|----------|--------------|----------|
| **Detección y conteo (40%)** | ✅ | Detecta y cuenta 6 objetos correctamente |
| **Tamaño del modelo (40%)** | ✅ | 16 MB optimizado con Float16 |
| **Aplicación web (15%)** | ✅ | Interfaz responsive, estable, drag & drop |
| **Documentación (5%)** | ✅ | README completo con detalles técnicos |

### Características de Entrega

- ✅ Carpeta `inventario/` con `index.html` funcional
- ✅ Bounding boxes en **color azul**
- ✅ **Números de clase** (0-5) en cada detección
- ✅ Tabla de **conteo total** por categoría
- ✅ Ejecución **100% local** sin APIs
- ✅ Documentación completa en `inventario/README.md`

---

## 🎓 Aprendizajes

1. **Pseudo-labeling:** Genera datasets anotados sin etiquetado manual
2. **Transfer Learning:** Reduce tiempo y mejora precisión
3. **TFLite en Web:** ML completamente offline en navegador
4. **NMS:** Esencial para eliminar duplicados en sliding window
5. **Optimización:** Float16 reduce tamaño sin pérdida significativa

---

## 📖 Referencias

- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [TensorFlow.js TFLite](https://github.com/tensorflow/tfjs/tree/master/tfjs-tflite)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [SSD: Single Shot Detector](https://arxiv.org/abs/1512.02325)

---

## 📄 Licencia

Proyecto académico desarrollado para el curso **BIG DATA - Módulo de Redes Convolucionales**.

**Estudiante:** Sebastian Forero (20242595001)
**Profesor:** Gerardo Muñoz
**Maestría:** Ciencias de la Computación y las Comunicaciones

---

**Fecha de entrega:** 9 de Noviembre 2024

**Estado:** ✅ APLICACIÓN WEB FUNCIONANDO

Ver documentación completa en: `inventario/README.md`
