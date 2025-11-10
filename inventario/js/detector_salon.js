// Detector SSD MobileNetV2 Custom - 20242595001 Sebastian Forero
// Modelo entrenado específicamente en objetos del salón

let model = null;
let uploadedImage = null;
const CLASSES = ['cpu', 'mesa', 'mouse', 'pantalla', 'silla', 'teclado'];
const INPUT_SIZE = 320;  // El modelo espera 320x320
const CONFIDENCE_THRESHOLD = 0.25;  // Threshold de confidence (bajado para testing)
const NMS_THRESHOLD = 0.45;

// Configuración de sliding window para 320x320
// Optimizado para imágenes grandes: menos ventanas, procesamiento más rápido
const WINDOW_SIZES = [320, 384];  // 2 tamaños (reducido de 3)
const STRIDE = 160;  // Stride aumentado para ~6x menos ventanas

// Cargar modelo al inicio
window.addEventListener('load', async () => {
    await loadModel();
    setupDragAndDrop();
});

// Cargar modelo TFLite
async function loadModel() {
    const statusDiv = document.getElementById('status');

    try {
        statusDiv.innerHTML = '<div class="status-message info"><span class="loading-spinner"></span>Cargando modelo de detección...</div>';

        // Cargar modelo TFLite entrenado
        model = await tflite.loadTFLiteModel('models/detector_salon.tflite');

        statusDiv.innerHTML = '<div class="status-message success">Modelo cargado correctamente. Sistema listo para detectar objetos.</div>';

        console.log('✅ Modelo SSD MobileNetV2 Custom cargado');
        console.log('Input: 320x320x3');
        console.log('Outputs: [bbox(4), class(6), confidence(1)]');
        console.log('Clases:', CLASSES);

    } catch (error) {
        console.error('Error cargando modelo:', error);
        statusDiv.innerHTML = '<div class="status-message error">Error al cargar el modelo. Verifique que el archivo models/detector_salon.tflite existe y que está ejecutando desde un servidor web (http://localhost:8000).</div>';
    }
}

// Configurar drag and drop
function setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

// Manejar selección de archivo
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Procesar archivo de imagen
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Por favor selecciona un archivo de imagen válido');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            uploadedImage = img;
            displayOriginalImage(img);
            document.getElementById('detectBtn').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Mostrar imagen original
function displayOriginalImage(img) {
    const canvas = document.getElementById('originalCanvas');
    const ctx = canvas.getContext('2d');

    const maxWidth = 550;
    const scale = maxWidth / img.width;
    canvas.width = maxWidth;
    canvas.height = img.height * scale;

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}

// Función principal de detección
async function detectObjects() {
    if (!model) {
        alert('El modelo aún no está cargado. Por favor espera...');
        return;
    }

    if (!uploadedImage) {
        alert('Por favor sube una imagen primero');
        return;
    }

    const statusDiv = document.getElementById('status');
    const detectBtn = document.getElementById('detectBtn');

    detectBtn.disabled = true;
    statusDiv.innerHTML = '<div class="status-message info"><span class="loading-spinner"></span>Procesando imagen y detectando objetos...</div>';

    try {
        console.log('🔍 Iniciando detección con modelo personalizado...');

        // Realizar detección con sliding window
        const detections = await slidingWindowDetection(uploadedImage);

        console.log(`\n📦 Detecciones brutas: ${detections.length}`);

        // Aplicar Non-Maximum Suppression
        const filteredDetections = applyNMS(detections);

        console.log(`✨ Detecciones después de NMS: ${filteredDetections.length}`);

        if (filteredDetections.length > 0) {
            console.log('Detecciones finales:');
            filteredDetections.forEach((det, i) => {
                console.log(`  ${i+1}. ${det.className} (${(det.confidence*100).toFixed(1)}%) en [${det.bbox.map(v => v.toFixed(2)).join(', ')}]`);
            });
        } else {
            console.warn('⚠️  NO SE ENCONTRARON OBJETOS');
        }

        // Dibujar resultados
        drawDetections(uploadedImage, filteredDetections);

        // Mostrar conteo
        displayCounts(filteredDetections);

        // Mostrar resultados
        document.getElementById('resultsSection').style.display = 'block';
        statusDiv.innerHTML = `<div class="status-message success">Detección completada. Total de objetos detectados: ${filteredDetections.length}</div>`;

    } catch (error) {
        console.error('Error en detección:', error);
        console.error('Stack:', error.stack);
        statusDiv.innerHTML = `<div class="status-message error">Error durante la detección: ${error.message}</div>`;
    } finally {
        detectBtn.disabled = false;
    }
}

// Sliding Window Detection con el modelo personalizado
async function slidingWindowDetection(img) {
    const detections = [];
    let totalWindows = 0;
    let windowsAboveThreshold = 0;

    // OPTIMIZACIÓN: Todas las imágenes se redimensionan a 320×240
    // Esto genera solo 1-2 ventanas para máxima velocidad
    const PROCESS_WIDTH = 320;
    const PROCESS_HEIGHT = 240;

    const originalWidth = img.width;
    const originalHeight = img.height;
    const scaleX = originalWidth / PROCESS_WIDTH;
    const scaleY = originalHeight / PROCESS_HEIGHT;

    console.log(`🔍 Imagen original: ${originalWidth}x${originalHeight}`);
    console.log(`   Redimensionando a: ${PROCESS_WIDTH}x${PROCESS_HEIGHT}`);

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    canvas.width = PROCESS_WIDTH;
    canvas.height = PROCESS_HEIGHT;
    ctx.drawImage(img, 0, 0, PROCESS_WIDTH, PROCESS_HEIGHT);

    console.log(`   Window sizes: [${WINDOW_SIZES}], Stride: ${STRIDE}`);
    console.log(`   Confidence threshold: ${CONFIDENCE_THRESHOLD}`);

    // Si la imagen procesada es más pequeña que la ventana mínima, procesar la imagen completa
    const minWindowSize = Math.min(...WINDOW_SIZES);
    if (PROCESS_WIDTH < minWindowSize || PROCESS_HEIGHT < minWindowSize) {
        console.log(`⚠️  Imagen pequeña (${PROCESS_WIDTH}x${PROCESS_HEIGHT}), procesando imagen completa...`);

        // Procesar imagen completa
        const imageData = ctx.getImageData(0, 0, PROCESS_WIDTH, PROCESS_HEIGHT);
        const resizedData = await resizeImageData(imageData, INPUT_SIZE, INPUT_SIZE);

        const outputs = tf.tidy(() => {
            let imgTensor = tf.browser.fromPixels(resizedData);
            imgTensor = tf.cast(imgTensor, 'float32');
            imgTensor = tf.div(imgTensor, 255.0);
            imgTensor = tf.expandDims(imgTensor, 0);
            return model.predict(imgTensor);
        });

        console.log(`   Outputs del modelo:`, outputs);
        console.log(`   Tipo de outputs:`, typeof outputs);
        console.log(`   Es array?:`, Array.isArray(outputs));

        // Extraer tensores por nombre
        // El modelo retorna un objeto con nombres:
        // StatefulPartitionedCall_1:1 → bbox [1, 4]
        // StatefulPartitionedCall_1:0 → class [1, 6]
        // StatefulPartitionedCall_1:2 → confidence [1, 1]

        let bboxTensor, classTensor, confidenceTensor;

        if (typeof outputs === 'object' && !Array.isArray(outputs)) {
            // Es un objeto/diccionario con nombres
            const keys = Object.keys(outputs);
            console.log(`   Keys de outputs:`, keys);

            // Buscar por tamaño de shape para identificar cada output
            for (const key of keys) {
                const tensor = outputs[key];
                const size = tensor.size;
                console.log(`   ${key}: shape=${tensor.shape}, size=${size}`);

                if (size === 4) {
                    // BBox [1, 4]
                    bboxTensor = tensor;
                    console.log(`   → Identificado como BBOX`);
                } else if (size === 6) {
                    // Class probs [1, 6]
                    classTensor = tensor;
                    console.log(`   → Identificado como CLASS`);
                } else if (size === 1) {
                    // Confidence [1, 1]
                    confidenceTensor = tensor;
                    console.log(`   → Identificado como CONFIDENCE`);
                }
            }
        } else if (Array.isArray(outputs)) {
            // Si es array
            bboxTensor = outputs[0];
            classTensor = outputs[1];
            confidenceTensor = outputs[2];
        } else {
            throw new Error('Formato de salida del modelo no soportado');
        }

        if (!bboxTensor || !classTensor || !confidenceTensor) {
            throw new Error('No se pudieron identificar todos los outputs del modelo');
        }

        const bboxData = await bboxTensor.data();
        const classProbs = await classTensor.data();
        const confidenceData = await confidenceTensor.data();

        // Limpiar tensores
        if (typeof outputs === 'object' && !Array.isArray(outputs)) {
            Object.values(outputs).forEach(t => t.dispose());
        } else if (Array.isArray(outputs)) {
            outputs.forEach(t => t.dispose());
        }

        const confidence = confidenceData[0];
        const classIdx = classProbs.indexOf(Math.max(...classProbs));
        const [ymin, xmin, ymax, xmax] = bboxData;

        console.log(`   Predicción del modelo:`);
        console.log(`   - Confidence: ${(confidence*100).toFixed(1)}%`);
        console.log(`   - Clase: ${CLASSES[classIdx]} (${classIdx})`);
        console.log(`   - Class probs: [${Array.from(classProbs).map((p, i) => `${CLASSES[i]}:${(p*100).toFixed(1)}%`).join(', ')}]`);
        console.log(`   - BBox: [${Array.from(bboxData).map(v => v.toFixed(3)).join(', ')}]`);

        if (confidence > CONFIDENCE_THRESHOLD) {
            // Convertir bbox a coordenadas de imagen ORIGINAL (escalar de vuelta)
            const bbox_x = xmin * PROCESS_WIDTH * scaleX;
            const bbox_y = ymin * PROCESS_HEIGHT * scaleY;
            const bbox_w = (xmax - xmin) * PROCESS_WIDTH * scaleX;
            const bbox_h = (ymax - ymin) * PROCESS_HEIGHT * scaleY;

            detections.push({
                x: bbox_x,
                y: bbox_y,
                width: bbox_w,
                height: bbox_h,
                class: classIdx,
                className: CLASSES[classIdx],
                confidence: confidence,
                classConfidence: classProbs[classIdx],
                bbox: [ymin, xmin, ymax, xmax]
            });

            console.log(`   ✅ DETECTADO (sobre threshold)`);
        } else {
            console.log(`   ❌ Descartado (confidence < ${CONFIDENCE_THRESHOLD})`);
        }

        return detections;
    }

    // Sliding window para imágenes (ahora todas son 640×480)
    console.log(`   Usando sliding window...`);

    for (const windowSize of WINDOW_SIZES) {
        // Solo usar ventanas que quepan en la imagen procesada
        if (windowSize > PROCESS_WIDTH || windowSize > PROCESS_HEIGHT) {
            console.log(`   ⏭️  Saltando ventana ${windowSize}x${windowSize} (no cabe)`);
            continue;
        }

        const scale = windowSize / INPUT_SIZE;

        for (let y = 0; y <= PROCESS_HEIGHT - windowSize; y += STRIDE) {
            for (let x = 0; x <= PROCESS_WIDTH - windowSize; x += STRIDE) {
                totalWindows++;

                // Extraer región
                const imageData = ctx.getImageData(x, y, windowSize, windowSize);

                // Redimensionar a INPUT_SIZE (320x320)
                const resizedData = await resizeImageData(imageData, INPUT_SIZE, INPUT_SIZE);

                // Convertir a tensor y predecir
                const outputs = tf.tidy(() => {
                    let imgTensor = tf.browser.fromPixels(resizedData);
                    imgTensor = tf.cast(imgTensor, 'float32');
                    imgTensor = tf.div(imgTensor, 255.0);
                    imgTensor = tf.expandDims(imgTensor, 0);
                    return model.predict(imgTensor);
                });

                // Extraer tensores (mismo formato que antes)
                let bboxTensor, classTensor, confidenceTensor;

                if (typeof outputs === 'object' && !Array.isArray(outputs)) {
                    // Es un objeto/diccionario
                    for (const key of Object.keys(outputs)) {
                        const tensor = outputs[key];
                        const size = tensor.size;

                        if (size === 4) bboxTensor = tensor;
                        else if (size === 6) classTensor = tensor;
                        else if (size === 1) confidenceTensor = tensor;
                    }
                } else if (Array.isArray(outputs)) {
                    bboxTensor = outputs[0];
                    classTensor = outputs[1];
                    confidenceTensor = outputs[2];
                }

                const bboxData = await bboxTensor.data();
                const classProbs = await classTensor.data();
                const confidenceData = await confidenceTensor.data();

                // Limpiar tensores
                if (typeof outputs === 'object' && !Array.isArray(outputs)) {
                    Object.values(outputs).forEach(t => t.dispose());
                } else if (Array.isArray(outputs)) {
                    outputs.forEach(t => t.dispose());
                }

                const confidence = confidenceData[0];

                // Debug primeras ventanas
                if (totalWindows <= 3) {
                    const classIdx = classProbs.indexOf(Math.max(...classProbs));
                    console.log(`   Ventana ${totalWindows}: pos(${x},${y}) size=${windowSize}`);
                    console.log(`   → ${CLASSES[classIdx]} conf=${(confidence*100).toFixed(1)}%`);
                    console.log(`   → bbox: [${Array.from(bboxData).map(v => v.toFixed(3)).join(', ')}]`);
                }

                // Si hay suficiente confianza
                if (confidence > CONFIDENCE_THRESHOLD) {
                    windowsAboveThreshold++;

                    // Obtener clase predicha
                    const classIdx = classProbs.indexOf(Math.max(...classProbs));
                    const classConfidence = classProbs[classIdx];

                    // Bbox predicho (normalizado 0-1 relativo a la ventana)
                    const [ymin, xmin, ymax, xmax] = bboxData;

                    // Convertir bbox a coordenadas de imagen procesada (640x480)
                    const bbox_x_proc = x + xmin * windowSize;
                    const bbox_y_proc = y + ymin * windowSize;
                    const bbox_w_proc = (xmax - xmin) * windowSize;
                    const bbox_h_proc = (ymax - ymin) * windowSize;

                    // Escalar a coordenadas de imagen ORIGINAL
                    const bbox_x = bbox_x_proc * scaleX;
                    const bbox_y = bbox_y_proc * scaleY;
                    const bbox_w = bbox_w_proc * scaleX;
                    const bbox_h = bbox_h_proc * scaleY;

                    detections.push({
                        x: bbox_x,
                        y: bbox_y,
                        width: bbox_w,
                        height: bbox_h,
                        class: classIdx,
                        className: CLASSES[classIdx],
                        confidence: confidence,
                        classConfidence: classConfidence,
                        bbox: [ymin, xmin, ymax, xmax],  // Bbox normalizado original
                        windowPos: {x, y, size: windowSize}
                    });

                    if (windowsAboveThreshold <= 5) {
                        console.log(`   ✅ Detección ${windowsAboveThreshold}: ${CLASSES[classIdx]} conf=${(confidence*100).toFixed(1)}%`);
                    }
                }
            }
        }
    }

    console.log(`📊 Total ventanas: ${totalWindows}`);
    console.log(`📊 Sobre threshold: ${windowsAboveThreshold}`);

    return detections;
}

// Redimensionar ImageData
async function resizeImageData(imageData, targetWidth, targetHeight) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = imageData.width;
    tmpCanvas.height = imageData.height;
    tmpCanvas.getContext('2d', { willReadFrequently: true }).putImageData(imageData, 0, 0);

    canvas.width = targetWidth;
    canvas.height = targetHeight;
    ctx.drawImage(tmpCanvas, 0, 0, targetWidth, targetHeight);

    return ctx.getImageData(0, 0, targetWidth, targetHeight);
}

// Non-Maximum Suppression
function applyNMS(detections) {
    if (detections.length === 0) return [];

    // Ordenar por confianza descendente
    detections.sort((a, b) => b.confidence - a.confidence);

    const keep = [];
    const suppressed = new Set();

    for (let i = 0; i < detections.length; i++) {
        if (suppressed.has(i)) continue;

        keep.push(detections[i]);

        for (let j = i + 1; j < detections.length; j++) {
            if (suppressed.has(j)) continue;

            const iou = calculateIoU(detections[i], detections[j]);

            if (iou > NMS_THRESHOLD) {
                suppressed.add(j);
            }
        }
    }

    return keep;
}

// Calcular Intersection over Union
function calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;
    const unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
}

// Dibujar detecciones en la imagen
function drawDetections(img, detections) {
    const canvas = document.getElementById('resultCanvas');
    const ctx = canvas.getContext('2d');

    const maxWidth = 550;
    const scale = maxWidth / img.width;
    canvas.width = maxWidth;
    canvas.height = img.height * scale;

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    detections.forEach(det => {
        const x = det.x * scale;
        const y = det.y * scale;
        const w = det.width * scale;
        const h = det.height * scale;

        // Bounding box en AZUL (#0052cc)
        ctx.strokeStyle = '#0052cc';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Etiqueta con clase y número
        const label = `${det.class}`;
        ctx.fillStyle = '#0052cc';
        ctx.font = 'bold 20px Arial';
        const textWidth = ctx.measureText(label).width;

        // Fondo para el texto
        ctx.fillRect(x, y - 28, textWidth + 12, 28);

        // Texto en blanco
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + 6, y - 8);
    });
}

// Mostrar conteo de objetos
function displayCounts(detections) {
    const counts = {};

    CLASSES.forEach((_, idx) => {
        counts[idx] = 0;
    });

    detections.forEach(det => {
        counts[det.class]++;
    });

    const tableBody = document.getElementById('countsTable');
    tableBody.innerHTML = '';

    let totalCount = 0;

    CLASSES.forEach((className, idx) => {
        const count = counts[idx];
        totalCount += count;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${idx}</strong></td>
            <td>${className.charAt(0).toUpperCase() + className.slice(1)}</td>
            <td><span class="count-value">${count}</span></td>
        `;
        tableBody.appendChild(row);
    });

    const totalRow = document.createElement('tr');
    totalRow.innerHTML = `
        <td colspan="2"><strong>TOTAL</strong></td>
        <td><span class="count-value" style="background: #28a745;">${totalCount}</span></td>
    `;
    tableBody.appendChild(totalRow);
}
