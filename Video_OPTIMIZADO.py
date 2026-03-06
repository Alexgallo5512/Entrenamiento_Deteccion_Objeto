#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SCRIPT OPTIMIZADO PARA iCAM-540 + YOLO
Basado en Video_10_YOlO.py con mejoras de rendimiento
"""

import cv2
import numpy as np
import time
import threading
import os
from pathlib import Path
from ultralytics import YOLO

# ================= CONFIGURACIÓN =================
MODEL_PATH = "best.pt"

# ✅ Rutas Compatible Windows/Linux
CAPTURAS_DIR = Path("capturas")
CAPTURAS_DIR.mkdir(exist_ok=True)
SAVE_PATH = CAPTURAS_DIR / "captura_sin_tapa.jpg"

# Resolución cámara (reducida para mejor rendimiento)
WIDTH  = 1280   # Antes: 1920 (- 33% = más rápido)
HEIGHT = 720    # Antes: 1080 (- 33% = más rápido)

# Tamaño YOLO (pequeño = procesamiento rápido)
YOLO_SIZE = 480 
YOLO_CONF = 0.5  # Confianza mínima (> 0.5 = más rápido)

# FPS objetivo
TARGET_FPS = 30

# ================= VARIABLES GLOBALES =================
model = YOLO(MODEL_PATH)

frame_lock = threading.Lock()
latest_frame = None
detection_event = threading.Event()

# Contadores FPS
cam_fps = 0
yolo_fps = 0
cam_count = 0
yolo_count = 0
last_fps_time = time.time()

icam_color = 0

# ================= FUNCIONES =================

def gst_to_opencv(sample):
    """Convierte formato SDK (GST) a formato OpenCV"""
    buf = sample.get_buffer()
    data = buf.extract_dup(0, buf.get_size())
    arr = np.frombuffer(data, dtype=np.uint8)

    if icam_color == 1:
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def new_image_handler(sample):
    """
    Callback de CÁMARA (debe ser LIVIANO, sin YOLO aquí)
    Solo captura frames, procesamiento en main loop
    """
    global latest_frame, cam_count

    if sample is None:
        return

    try:
        frame = gst_to_opencv(sample)
        
        with frame_lock:
            latest_frame = frame

        cam_count += 1
        
    except Exception as e:
        print(f"❌ Error en callback cámara: {e}")


def save_detection(frame, class_name="Sin_Tapa"):
    """Guarda imagen de detección"""
    try:
        cv2.imwrite(str(SAVE_PATH), frame)
        print(f"✅ Detección guardada: {SAVE_PATH}")
        detection_event.set()
    except Exception as e:
        print(f"❌ Error al guardar: {e}")


# ================= MAIN =================
if __name__ == "__main__":
    print("🚀 Iniciando iCAM-540 + YOLO con threading...")
    
    try:
        from CamNavi2 import CamNavi2
        try:
            cn2 = CamNavi2.CamNavi2()
        except:
            cn2 = CamNavi2()
    except ImportError:
        print("❌ Error: No se encontró CamNavi2. Instala el SDK de iCAM-540")
        exit(1)

    try:
        # Obtener cámara
        camera_list = cn2.enum_camera_list()
        if not camera_list:
            print("❌ No se encontró cámara iCAM")
            exit(1)
        
        camera = cn2.get_device_by_name("iCam500")  # O "iCam540" si es 540
        print(f"✅ Cámara detectada: {camera}")

        # Verificar color/mono
        icam_color = int(cn2.advcam_query_fw_sku(camera))
        print(f"📷 Modo: {'Color' if icam_color == 1 else 'Mono'}")

        # ========== CONFIGURAR PIPELINE ==========
        pipe_params = {
            "acq_mode": 0,
            "width": WIDTH,
            "height": HEIGHT,
            "enable_infer": 0  # IMPORTANTE: sin inferencia en cámara
        }

        if icam_color == 1:
            pipe_params["format"] = "YUY2"

        cn2.advcam_config_pipeline(camera, **pipe_params)
        cn2.advcam_open(camera, -1)
        cn2.advcam_register_new_image_handler(camera, new_image_handler)

        # ========== CONFIGURACIÓN ÓPTICA ==========
        camera.lighting.selector = 2
        camera.lighting.gain = 100
        cn2.advcam_set_img_sharpness(camera, 90)
        cn2.advcam_set_img_brightness(camera, 50)
        cn2.advcam_set_img_gain(camera, 5)
        camera.set_acq_frame_rate(TARGET_FPS)

        # ========== INICIAR CAPTURA ==========
        cn2.advcam_play(camera)
        camera.focus.pos_zero()
        time.sleep(0.5)
        camera.focus.direction = 0

        print("▶️  Sistema iniciado. Presiona ESC para salir")
        print(f"   📊 Resolución cámara: {WIDTH}x{HEIGHT}")
        print(f"   🧠 Tamaño YOLO: {YOLO_SIZE}x{YOLO_SIZE}")
        print(f"   ⚙️  Confianza YOLO: {YOLO_CONF}")
        print()

        # ================= LOOP PRINCIPAL =================
        while True:
            # Obtener frame más reciente (con lock)
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)  # Esperar frame
                    continue
                frame = latest_frame.copy()

            # -------- REDIMENSIONAR PARA YOLO --------
            # IMPORTANTE: esto acelera mucho la inferencia
            resized = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
            
            # -------- INFERENCIA YOLO --------
            results = model(
                resized, 
                verbose=False,
                conf=YOLO_CONF,  # Confianza mínima = más rápido
                device=0  # GPU si disponible
            )

            yolo_count += 1

            # -------- DIBUJAR RESULTADOS --------
            annotated = results[0].plot()

            # -------- LÓGICA DE DETECCIÓN --------
            detected = False
            for cls_id in results[0].boxes.cls.tolist():
                class_name = results[0].names[int(cls_id)]
                if class_name == "Sin_Tapa":
                    print(f"🎯 Objeto detectado: {class_name}")
                    save_detection(annotated, class_name)
                    detected = True
                    break

            # -------- MOSTRAR EN PANTALLA --------
            # Agregar info de FPS
            info_text = f"CAM: {cam_fps} FPS | YOLO: {yolo_fps} FPS"
            cv2.putText(
                annotated, 
                info_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("iCAM-540 + YOLO (Optimizado)", annotated)

            # -------- CONTAR FPS --------
            now = time.time()
            if now - last_fps_time >= 1.0:
                cam_fps = cam_count
                yolo_fps = yolo_count
                cam_count = 0
                yolo_count = 0
                last_fps_time = now
                print(f"📊 FPS → Cámara: {cam_fps} | YOLO: {yolo_fps}")

            # -------- CONTROLES TECLADO --------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n⏹️  Deteniendo...")
                break
            elif key == ord('+') or key == ord('='):  # Enfoque adelante
                try:
                    camera.focus.direction = 0
                    camera.focus.distance = 30
                    print("🔍 Enfoque: Adelante")
                except:
                    pass
            elif key == ord('-'):  # Enfoque atrás
                try:
                    camera.focus.direction = 1
                    camera.focus.distance = 30
                    print("🔍 Enfoque: Atrás")
                except:
                    pass
            elif key == ord('s'):  # Captura manual
                if latest_frame is not None:
                    save_detection(annotated, "Manual")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ========== LIMPIEZA ==========
        print("\n🧹 Limpiando recursos...")
        try:
            camera.lighting.selector = 0
            cn2.advcam_register_new_image_handler(camera, None)
            cn2.advcam_close(camera)
        except:
            pass
        cv2.destroyAllWindows()
        print("✅ Finalizado")
