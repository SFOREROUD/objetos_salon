"""
Generar etiquetas automáticas usando el modelo SSD preentrenado
Esto crea un dataset anotado para entrenar nuestro propio modelo
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import json
import urllib.request
import tarfile

print("="*70)
print("GENERACIÓN AUTOMÁTICA DE ETIQUETAS CON SSD COCO")
print("="*70)

# Configuración
MODEL_DIR = 'modelos_preentrenados'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
MODEL_URL = f'http://download.tensorflow.org/models/object_detection/tf2/20200711/{MODEL_NAME}.tar.gz'
DATA_DIR = 'C:/Users/Sebastian/Downloads/Photos-1-001/data/processed'
OUTPUT_DIR = 'C:/Users/Sebastian/Downloads/Photos-1-001/dataset_anotado'

# Mapeo de clases COCO a nuestras clases
COCO_TO_OURS = {
    62: 4,  # chair -> silla
    67: 1,  # dining table -> mesa
    72: 3,  # tv -> pantalla
    73: 0,  # laptop -> cpu (aproximación)
    74: 2,  # mouse -> mouse
    76: 5   # keyboard -> teclado
}

OUR_CLASSES = ['cpu', 'mesa', 'mouse', 'pantalla', 'silla', 'teclado']
CONFIDENCE_THRESHOLD = 0.30

# Crear directorios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

print(f"\n1. Descargando/Verificando modelo SSD preentrenado...")
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, MODEL_NAME)

if not os.path.exists(model_path):
    print(f"   Descargando desde TensorFlow Model Zoo...")
    print(f"   URL: {MODEL_URL}")
    tar_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.tar.gz')

    # Descargar
    urllib.request.urlretrieve(MODEL_URL, tar_path)

    # Extraer
    print(f"   Extrayendo modelo...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(MODEL_DIR)

    # Limpiar
    os.remove(tar_path)
    print(f"   ✅ Modelo descargado y extraído")
else:
    print(f"   ✅ Modelo ya existe")

print(f"\n2. Cargando modelo SSD preentrenado...")
saved_model_path = os.path.join(MODEL_DIR, MODEL_NAME, 'saved_model')
detect_fn = tf.saved_model.load(saved_model_path)
print("   ✅ Modelo cargado")

print(f"\n3. Procesando imágenes y generando etiquetas...")
print("="*70)

total_images = 0
total_detections = 0
images_with_detections = 0

for class_idx, class_name in enumerate(OUR_CLASSES):
    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.exists(class_path):
        print(f"⚠️  No se encontró {class_path}")
        continue

    image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

    print(f"\n📸 Procesando clase: {class_name} ({len(image_files)} imágenes)")
    print("-"*70)

    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)

        # Leer imagen
        image = cv2.imread(img_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convertir a tensor
        input_tensor = tf.convert_to_tensor(image_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Detectar
        detections = detect_fn(input_tensor)

        # Extraer resultados
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)

        # Generar anotaciones en formato YOLO
        annotations = []

        for i in range(len(scores)):
            if scores[i] > CONFIDENCE_THRESHOLD:
                class_id = classes[i]

                # Solo clases que nos interesan
                if class_id in COCO_TO_OURS:
                    our_class_id = COCO_TO_OURS[class_id]
                    box = boxes[i]  # [ymin, xmin, ymax, xmax] normalizado

                    # Convertir a formato YOLO: [class x_center y_center width height]
                    ymin, xmin, ymax, xmax = box
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2
                    box_width = xmax - xmin
                    box_height = ymax - ymin

                    annotations.append({
                        'class_id': our_class_id,
                        'class_name': OUR_CLASSES[our_class_id],
                        'bbox': [x_center, y_center, box_width, box_height],
                        'confidence': float(scores[i])
                    })

        # Guardar imagen y anotaciones
        if annotations:
            total_images += 1
            images_with_detections += 1
            total_detections += len(annotations)

            # Copiar imagen
            img_output_name = f"{class_name}_{total_images:04d}.jpg"
            img_output_path = os.path.join(OUTPUT_DIR, 'images', img_output_name)
            cv2.imwrite(img_output_path, image)

            # Guardar anotaciones en formato YOLO (.txt)
            txt_output_name = f"{class_name}_{total_images:04d}.txt"
            txt_output_path = os.path.join(OUTPUT_DIR, 'labels', txt_output_name)

            with open(txt_output_path, 'w') as f:
                for ann in annotations:
                    bbox = ann['bbox']
                    f.write(f"{ann['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

            # Mostrar progreso
            if total_images % 10 == 0:
                print(f"  Procesadas: {total_images} imágenes...")
        else:
            # Imagen sin detecciones
            total_images += 1

print(f"\n  ✅ Clase {class_name} completada")

print("\n" + "="*70)
print("📊 RESUMEN DE ETIQUETADO AUTOMÁTICO")
print("="*70)
print(f"Total de imágenes procesadas: {total_images}")
print(f"Imágenes con detecciones: {images_with_detections} ({images_with_detections/total_images*100:.1f}%)")
print(f"Total de detecciones (bounding boxes): {total_detections}")
print(f"Promedio de objetos por imagen: {total_detections/images_with_detections:.1f}")

print(f"\n📁 Archivos generados en: {OUTPUT_DIR}")
print(f"  - images/: {images_with_detections} imágenes")
print(f"  - labels/: {images_with_detections} archivos .txt (formato YOLO)")

# Crear archivo de configuración para YOLO
config = {
    'path': os.path.abspath(OUTPUT_DIR),
    'train': 'images',
    'val': 'images',
    'nc': len(OUR_CLASSES),
    'names': OUR_CLASSES
}

config_path = os.path.join(OUTPUT_DIR, 'dataset.yaml')
with open(config_path, 'w') as f:
    f.write(f"path: {config['path']}\n")
    f.write(f"train: {config['train']}\n")
    f.write(f"val: {config['val']}\n")
    f.write(f"nc: {config['nc']}\n")
    f.write(f"names: {config['names']}\n")

print(f"\n✅ Configuración YOLO guardada en: {config_path}")

print("\n" + "="*70)
print("🚀 PRÓXIMO PASO: ENTRENAR YOLOV8")
print("="*70)
print("Ahora puedes entrenar YOLOv8 con estas etiquetas automáticas:")
print("")
print("1. Instalar ultralytics:")
print("   pip install ultralytics")
print("")
print("2. Entrenar:")
print("   python entrenar_yolo.py")
print("")
print("Este modelo será MUCHO mejor porque:")
print("  ✅ Tiene bounding boxes precisos")
print("  ✅ Está entrenado específicamente en tus objetos")
print("  ✅ Se puede exportar fácilmente a TFLite")
print("="*70)

# Guardar estadísticas
stats = {
    'total_images': total_images,
    'images_with_detections': images_with_detections,
    'total_detections': total_detections,
    'classes': OUR_CLASSES,
    'threshold': CONFIDENCE_THRESHOLD
}

with open(os.path.join(OUTPUT_DIR, 'stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ Estadísticas guardadas en: {os.path.join(OUTPUT_DIR, 'stats.json')}")
