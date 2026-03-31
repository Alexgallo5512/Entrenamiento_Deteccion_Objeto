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
SAVE_PATH = "/home/icam-540/capturas/captura_sin_tapa.jpg"

# Resolución cámara (reducida para mejor rendimiento)
WIDTH  = 1280   
HEIGHT = 720   
# Tamaño YOLO (pequeño = procesamiento rápido)
YOLO_SIZE_W = 640 
YOLO_SIZE_H = 480 
YOLO_CONF = 0.3  # Confianza mínima (> 0.5 = más rápido)

# FPS objetivo
TARGET_FPS = 40

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

gain =1
sharpness =5
brightness =10
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
        camera.dio.do0.op_mode = 0
        camera.dio.do0.user_output = 0 # DO low, DI high
        print("DO lOW " + str(camera.dio.do0.user_output))
        camera.dio.do0.reverse = 0
        # ========== CONFIGURACIÓN ÓPTICA ==========
        camera.lighting.selector = 2
        camera.lighting.gain = 40

        camera.image.saturation = 119
        camera.image.gamma = 24
        
        cn2.advcam_set_img_sharpness(camera, 15)
        cn2.advcam_set_img_brightness(camera, 80)
        cn2.advcam_set_img_gain(camera, 6)

        camera.set_acq_frame_rate(TARGET_FPS)

        # ========== INICIAR CAPTURA ==========
        
        camera.focus.pos_zero()
        time.sleep(0.5)
        camera.focus.distance = 65
        i = 0
        while i < 7:
            
            camera.focus.direction = 1 # lens focusing motor backward
            try:
                camera.focus.distance = 100
                print("lens motor posistion: ", camera.focus.position())
                i+=1
                print("valor ", i)
            except ValueError:
                print("lens position out of index")
        
        cn2.advcam_play(camera)
        print("▶️  Sistema iniciado. Presiona ESC para salir")
        print(f"   📊 Resolución cámara: {WIDTH}x{HEIGHT}")
        print(f"   🧠 Tamaño YOLO: {YOLO_SIZE_W}x{YOLO_SIZE_H}")
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
            resized = cv2.resize(frame, (YOLO_SIZE_W, YOLO_SIZE_H))
            
            # -------- INFERENCIA YOLO --------
            results = model(
               resized, 
                verbose=False,
                conf=YOLO_CONF  # Confianza mínima = más rápido
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
            elif key == ord('a') or key == ord('A'):  # Enfoque RETROCEDE
                try:
                    camera.focus.direction = 0
                    camera.focus.distance = 5
                    print("lens motor Retrocede posistion: ", camera.focus.position())
                except:
                    pass
            elif key == ord('b') or key == ord('B'):  # Enfoque ADELANTA
                try:
                    camera.focus.direction = 1
                    camera.focus.distance = 5
                    print("lens motor Adelante posistion: ", camera.focus.position())
                except:
                    pass

                #GAIN
            elif key == ord('n') or key == ord('N'):  # GAIN AUMENTAR
                try:

                    gain_a = cn2.advcam_get_img_gain(camera)
                    gain_a = gain_a + gain
                    cn2.advcam_set_img_gain(camera, gain_a)
                    print("gain aumento: ", cn2.advcam_get_img_gain(camera))
                except Exception as e:
                    print(f"❌ Error gain  a: {e}")
                    pass
            elif key == ord('m') or key == ord('M'):  # GAIN DISMINUIR
                try:

                    gain_a = cn2.advcam_get_img_gain(camera)
                    gain_a = gain_a - gain
                    cn2.advcam_set_img_gain(camera, gain_a)
                    print("gain disminuir: ", cn2.advcam_get_img_gain(camera))
                except Exception as e:
                    print(f"❌ Error gain d: {e}")
                    pass

                #SHARPNESS
            elif key == ord('v') or key == ord('V'):  # SHARPNESS AUMENTAR
                try:

                    sharpness_a = cn2.advcam_get_img_sharpness(camera)
                    sharpness_a = sharpness_a + sharpness
                    cn2.advcam_set_img_sharpness(camera, sharpness_a)
                    print("sharpness aumento: ", cn2.advcam_get_img_sharpness(camera))
                except Exception as e:
                    print(f"❌ Error sharpness  a: {e}")
                    pass
            elif key == ord('c') or key == ord('C'):  # SHARPNESS DISMINUIR
                try:

                    sharpness_a = cn2.advcam_get_img_sharpness(camera)
                    sharpness_a = sharpness_a - sharpness
                    cn2.advcam_set_img_sharpness(camera, sharpness_a)
                    print("sharpness disminuir: ", cn2.advcam_get_img_sharpness(camera))
                except Exception as e:
                    print(f"❌ Error sharpness d: {e}")
                    pass



                #BRIGTHNESS
            elif key == ord('x') or key == ord('X'):  # BRIGTHNESS AUMENTAR
                try:

                    brightness_a = cn2.advcam_get_img_brightness(camera)
                    brightness_a = brightness_a + brightness
                    cn2.advcam_set_img_brightness(camera, brightness_a)
                    print("brightness aumento: ", cn2.advcam_get_img_brightness(camera))
                except Exception as e:
                    print(f"❌ Error brightness  a: {e}")
                    pass
            elif key == ord('z') or key == ord('X'):  # BRIGTHNESS DISMINUIR
                try:

                    brightness_a = cn2.advcam_get_img_brightness(camera)
                    brightness_a = brightness_a - brightness
                    cn2.advcam_set_img_brightness(camera, brightness_a)
                    print("brightness disminuir: ", cn2.advcam_get_img_brightness(camera))
                except Exception as e:
                    print(f"❌ Error brightness d: {e}")
                    pass    
            elif key == ord('s'):  # Captura manual
                if latest_frame is not None:
                    save_detection(annotated, "Manual")
            elif key == ord('t'):  
                    
                    camera.dio.do0.user_output = 1
                    salida  = str(camera.dio.do0.user_output)
                    print("DO high " + salida)
                    #camera.dio.do0.user_output = 1 # DO high, DI low
                    level =  camera.dio.di0.level
                    print(level)
            elif key == ord('r'):  
                    camera.dio.do0.user_output = 0
                    salida  = str(camera.dio.do0.user_output)
                    print("DO low " + salida)
                    #camera.dio.do0.user_output = 1 # DO high, DI low
                    level =  camera.dio.di0.level
                    print(level)

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
