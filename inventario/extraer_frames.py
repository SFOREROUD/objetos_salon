"""
Extraer frames de los videos del salón para crear el dataset
"""

import cv2
import os
from pathlib import Path

print("="*70)
print("EXTRACCIÓN DE FRAMES DE VIDEOS")
print("="*70)

# Configuración
VIDEO_DIR = '.'
OUTPUT_DIR = 'data/processed'
FRAME_INTERVAL = 30  # Extraer 1 frame cada 30 frames (~1 por segundo a 30fps)
CODIGO = '20242595001'

# Mapeo de videos a clases (manual según el contenido)
VIDEO_TO_CLASS = {
    'VID_20251029_194600.mp4': 'cpu',
    'VID_20251029_194608.mp4': 'mesa',
    'VID_20251029_194621.mp4': 'mouse',
    'VID_20251029_194628.mp4': 'pantalla',
    'VID_20251029_194641.mp4': 'silla',
    'VID_20251029_194655.mp4': 'teclado',
    'VID_20251029_194701.mp4': 'general'  # Mix de objetos
}

# Crear directorios de salida
for class_name in ['cpu', 'mesa', 'mouse', 'pantalla', 'silla', 'teclado']:
    os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

print(f"\n1. Buscando videos en: {VIDEO_DIR}")

# Buscar todos los videos
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
print(f"   Encontrados {len(video_files)} videos")

total_frames = 0

for video_file in sorted(video_files):
    video_path = os.path.join(VIDEO_DIR, video_file)

    # Determinar clase (si está en el mapeo, sino distribuir uniformemente)
    if video_file in VIDEO_TO_CLASS:
        class_name = VIDEO_TO_CLASS[video_file]
    else:
        class_name = 'general'

    print(f"\n📹 Procesando: {video_file}")
    print(f"   Clase asignada: {class_name}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"   ❌ Error abriendo video")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"   FPS: {fps:.1f}, Total frames: {total_video_frames}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Extraer 1 frame cada FRAME_INTERVAL
        if frame_count % FRAME_INTERVAL == 0:
            # Guardar frame
            output_path = os.path.join(OUTPUT_DIR, class_name, f"{CODIGO}_{total_frames:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            total_frames += 1

        frame_count += 1

    cap.release()
    print(f"   ✅ Extraídos {saved_count} frames")

print(f"\n" + "="*70)
print(f"✅ EXTRACCIÓN COMPLETADA")
print("="*70)
print(f"Total de frames extraídos: {total_frames}")
print(f"Directorio de salida: {OUTPUT_DIR}")

# Mostrar distribución por clase
print(f"\n📊 Distribución por clase:")
for class_name in ['cpu', 'mesa', 'mouse', 'pantalla', 'silla', 'teclado']:
    class_path = os.path.join(OUTPUT_DIR, class_name)
    count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
    print(f"   {class_name}: {count} imágenes")

print(f"\n🚀 Siguiente paso:")
print(f"   python generar_etiquetas_auto.py")
print("="*70)
