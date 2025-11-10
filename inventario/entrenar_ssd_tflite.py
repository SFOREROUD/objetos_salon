"""
Entrenar SSD MobileNetV2 con las etiquetas automáticas y exportar a TFLite
Usa TensorFlow Object Detection API para entrenar un detector optimizado
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

print("="*70)
print("ENTRENAMIENTO SSD MobileNetV2 PARA TFLite")
print("="*70)

# Configuración
DATASET_DIR = 'dataset_anotado'
MODEL_DIR = 'modelos_preentrenados'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
OUTPUT_DIR = 'modelo_entrenado_ssd'
TFRECORD_DIR = 'tfrecords'

INPUT_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.001

OUR_CLASSES = ['cpu', 'mesa', 'mouse', 'pantalla', 'silla', 'teclado']

print("\n📊 Configuración:")
print(f"  Input size: {INPUT_SIZE}x{INPUT_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")

# Leer estadísticas del dataset
with open(os.path.join(DATASET_DIR, 'stats.json'), 'r') as f:
    stats = json.load(f)

print(f"\n📁 Dataset:")
print(f"  Imágenes con detecciones: {stats['images_with_detections']}")
print(f"  Total de bounding boxes: {stats['total_detections']}")
print(f"  Promedio por imagen: {stats['total_detections']/stats['images_with_detections']:.1f}")

# Cargar todas las imágenes y anotaciones
print("\n1. Cargando dataset...")

images_dir = os.path.join(DATASET_DIR, 'images')
labels_dir = os.path.join(DATASET_DIR, 'labels')

dataset = []

for img_file in os.listdir(images_dir):
    if not img_file.endswith('.jpg'):
        continue

    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))

    if not os.path.exists(label_path):
        continue

    # Leer imagen para obtener dimensiones
    img = cv2.imread(img_path)
    if img is None:
        continue

    height, width = img.shape[:2]

    # Leer anotaciones YOLO
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:])

            # Convertir de YOLO (normalizado) a formato [ymin, xmin, ymax, xmax] (normalizado)
            xmin = x_center - box_width / 2
            ymin = y_center - box_height / 2
            xmax = x_center + box_width / 2
            ymax = y_center + box_height / 2

            boxes.append({
                'class_id': class_id,
                'bbox': [ymin, xmin, ymax, xmax]  # Formato TF
            })

    if boxes:
        dataset.append({
            'image_path': img_path,
            'width': width,
            'height': height,
            'boxes': boxes
        })

print(f"   ✅ Cargadas {len(dataset)} imágenes con anotaciones")

# Dividir en train/val
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

print(f"   Train: {len(train_data)} imágenes")
print(f"   Val: {len(val_data)} imágenes")

# Crear generador de datos
print("\n2. Creando generadores de datos...")

def load_batch(data_list, batch_size, augment=False):
    """Generador de batches para entrenamiento"""
    indices = np.arange(len(data_list))

    while True:
        np.random.shuffle(indices)

        for start_idx in range(0, len(data_list), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            images = []
            all_boxes = []
            all_classes = []

            for idx in batch_indices:
                item = data_list[idx]

                # Leer imagen
                img = cv2.imread(item['image_path'])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize a tamaño fijo
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

                # Normalizar
                img = img.astype(np.float32) / 255.0

                # Augmentación simple
                if augment and np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    # Invertir coordenadas x de los boxes
                    boxes_flipped = []
                    for box in item['boxes']:
                        ymin, xmin, ymax, xmax = box['bbox']
                        boxes_flipped.append({
                            'class_id': box['class_id'],
                            'bbox': [ymin, 1-xmax, ymax, 1-xmin]
                        })
                    boxes = boxes_flipped
                else:
                    boxes = item['boxes']

                images.append(img)

                # Extraer boxes y clases
                batch_boxes = []
                batch_classes = []
                for box in boxes:
                    batch_boxes.append(box['bbox'])
                    batch_classes.append(box['class_id'])

                all_boxes.append(batch_boxes)
                all_classes.append(batch_classes)

            yield {
                'images': np.array(images),
                'boxes': all_boxes,
                'classes': all_classes
            }

print("   ✅ Generadores creados")

# Construir modelo detector usando Keras Functional API
print("\n3. Construyendo modelo detector...")

def build_ssd_model():
    """
    Construir un modelo SSD simple usando MobileNetV2 como backbone
    """
    # Backbone: MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )

    # Congelar capas base inicialmente
    base_model.trainable = False

    # Input
    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))

    # Backbone
    x = base_model(inputs, training=False)

    # Detection head
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Outputs
    # Para simplificar, predecir un solo objeto por imagen (el más prominente)
    # Formato: [class_id, x_center, y_center, width, height, confidence]
    num_classes = len(OUR_CLASSES)

    # Clasificación
    class_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='class')(x)

    # Regresión de bounding box (4 valores: ymin, xmin, ymax, xmax normalizados)
    bbox_output = tf.keras.layers.Dense(4, activation='sigmoid', name='bbox')(x)

    # Confidence score
    conf_output = tf.keras.layers.Dense(1, activation='sigmoid', name='confidence')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[class_output, bbox_output, conf_output])

    return model, base_model

model, base_model = build_ssd_model()

print(f"   ✅ Modelo construido")
print(f"   Total parámetros: {model.count_params():,}")

model.summary()

# Compilar
print("\n4. Compilando modelo...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss={
        'class': 'sparse_categorical_crossentropy',
        'bbox': 'mse',
        'confidence': 'binary_crossentropy'
    },
    loss_weights={
        'class': 1.0,
        'bbox': 1.0,
        'confidence': 0.5
    },
    metrics={
        'class': 'accuracy'
    }
)

print("   ✅ Modelo compilado")

# Preparar datos para entrenamiento
print("\n5. Preparando datos de entrenamiento...")

def prepare_training_data(data_list):
    """Convertir datos a formato para entrenamiento"""
    X = []
    y_class = []
    y_bbox = []
    y_conf = []

    for item in data_list:
        # Leer imagen
        img = cv2.imread(item['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = img.astype(np.float32) / 255.0

        # Tomar el primer box como target (simplificación)
        if item['boxes']:
            first_box = item['boxes'][0]
            class_id = first_box['class_id']
            bbox = first_box['bbox']

            X.append(img)
            y_class.append(class_id)
            y_bbox.append(bbox)
            y_conf.append(1.0)  # Hay objeto presente

    return (
        np.array(X),
        {
            'class': np.array(y_class),
            'bbox': np.array(y_bbox),
            'confidence': np.array(y_conf)
        }
    )

X_train, y_train = prepare_training_data(train_data)
X_val, y_val = prepare_training_data(val_data)

print(f"   ✅ Datos preparados")
print(f"   Train: {X_train.shape[0]} muestras")
print(f"   Val: {X_val.shape[0]} muestras")

# Entrenar
print("\n6. Entrenando modelo...")
print("="*70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Fase 1: Entrenar solo el head
print("\n📌 FASE 1: Entrenando detection head (backbone congelado)")
history1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Fase 2: Fine-tuning completo
print("\n📌 FASE 2: Fine-tuning completo (descongelando backbone)")
base_model.trainable = True

# Recompilar con learning rate más bajo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss={
        'class': 'sparse_categorical_crossentropy',
        'bbox': 'mse',
        'confidence': 'binary_crossentropy'
    },
    loss_weights={
        'class': 1.0,
        'bbox': 1.0,
        'confidence': 0.5
    },
    metrics={
        'class': 'accuracy'}
)

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=70,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Entrenamiento completado")

# Guardar modelo
print("\n7. Guardando modelo...")

model.save(os.path.join(OUTPUT_DIR, 'modelo_final.keras'))
print(f"   ✅ Modelo Keras guardado")

# Convertir a TFLite
print("\n8. Convirtiendo a TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizaciones
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

tflite_path = os.path.join(OUTPUT_DIR, 'detector_salon.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)

print("="*70)
print("✅ MODELO TFLite GENERADO")
print("="*70)
print(f"Archivo: {tflite_path}")
print(f"Tamaño: {size_mb:.2f} MB")
print("="*70)

# Probar el modelo
print("\n9. Probando modelo TFLite...")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n📊 Detalles del modelo:")
print(f"  Input: {input_details[0]['shape']}")
print(f"  Outputs: {len(output_details)}")
for i, detail in enumerate(output_details):
    print(f"    {i+1}. {detail['name']}: {detail['shape']}")

# Probar con una imagen
test_item = val_data[0]
img = cv2.imread(test_item['image_path'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
img = img.astype(np.float32) / 255.0
img_input = np.expand_dims(img, axis=0)

interpreter.set_tensor(input_details[0]['index'], img_input)
interpreter.invoke()

class_output = interpreter.get_tensor(output_details[0]['index'])[0]
bbox_output = interpreter.get_tensor(output_details[1]['index'])[0]
conf_output = interpreter.get_tensor(output_details[2]['index'])[0]

predicted_class = np.argmax(class_output)
confidence = conf_output[0]

print(f"\n✅ Prueba de detección:")
print(f"  Clase predicha: {OUR_CLASSES[predicted_class]} (confianza: {confidence:.2%})")
print(f"  Bounding box: {bbox_output}")
print(f"  Ground truth: {test_item['boxes'][0]['class_id']} - {OUR_CLASSES[test_item['boxes'][0]['class_id']]}")

print("\n" + "="*70)
print("🚀 SIGUIENTE PASO: INTEGRAR EN LA WEB")
print("="*70)
print(f"Copia el archivo TFLite a tu carpeta web:")
print(f"  {tflite_path}")
print(f"  → inventario/models/detector_salon.tflite")
print("="*70)

# Guardar información del modelo
model_info = {
    'input_size': INPUT_SIZE,
    'classes': OUR_CLASSES,
    'size_mb': size_mb,
    'train_images': len(train_data),
    'val_images': len(val_data),
    'total_boxes': stats['total_detections']
}

with open(os.path.join(OUTPUT_DIR, 'model_info.json'), 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n✅ Información del modelo guardada en: {os.path.join(OUTPUT_DIR, 'model_info.json')}")
