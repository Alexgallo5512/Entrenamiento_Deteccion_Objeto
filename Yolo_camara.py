#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
from CamNavi2 import CamNavi2
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "best.pt"
SAVE_PATH = "/home/icam-540/capturas/captura_sin_tapa_2.jpg"

WIDTH  = 1920
HEIGHT = 1080 #Resolucion 
YOLO_SIZE = 600   # tamaño para inferencia
# ==========================================

model = YOLO(MODEL_PATH)

frame_lock = threading.Lock()
latest_frame = None


# FPS counters
cam_fps = 0
yolo_fps = 0
cam_count = 0
yolo_count = 0
last_fps_time = time.time()

icam_color = 0


# ---------- SDK → OpenCV ----------
def gst_to_opencv(sample):
    buf = sample.get_buffer()
    data = buf.extract_dup(0, buf.get_size())
    arr = np.frombuffer(data, dtype=np.uint8)

    if icam_color == 1:
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


# ---------- CALLBACK CÁMARA (LIVIANO) ----------
def new_image_handler(sample):
    global latest_frame, cam_count

    if sample is None:
        return

    frame = gst_to_opencv(sample)

    with frame_lock:
        latest_frame = frame

    cam_count += 1


# ================= MAIN =================
if __name__ == "__main__":

    try:
        cn2 = CamNavi2.CamNavi2()
    except:
        cn2 = CamNavi2()

    camera_list = cn2.enum_camera_list()
    camera = cn2.get_device_by_name("iCam500")  # ajusta si es iCam540

    icam_color = int(cn2.advcam_query_fw_sku(camera))

    # ---------- PIPELINE ----------
    pipe_params = {
        "acq_mode": 0,
        "width": WIDTH,
        "height": HEIGHT,
        "enable_infer": 0
    }

    if icam_color == 1:
        pipe_params["format"] = "YUY2"

    cn2.advcam_config_pipeline(camera, **pipe_params)
    cn2.advcam_open(camera, -1)

    cn2.advcam_register_new_image_handler(camera, new_image_handler)

    # ---------- CÁMARA SETTINGS ----------
    camera.lighting.selector = 2
    camera.lighting.gain = 100
    cn2.advcam_set_img_sharpness(camera, 90)
    cn2.advcam_set_img_brightness(camera, 50)
    cn2.advcam_set_img_gain(camera, 5)
    camera.set_acq_frame_rate(30)

    cn2.advcam_play(camera)
    camera.focus.pos_zero()
    time.sleep(0.5)
    camera.focus.direction = 0
    print(camera.focus.distance)
    
    print("▶ Presiona ESC para salir")

    # ================= LOOP PRINCIPAL =================
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        
        # -------- YOLO --------
        resized = cv2.resize(frame, (480, 480))
        results = model(resized, verbose=False)

        yolo_count += 1

         #if len(results[0].boxes) > 0:
        annotated = results[0].plot()
         # else:
           # continue
        # -------- DETECCIÓN --------
        for cls_id in results[0].boxes.cls.tolist():
            name = results[0].names[int(cls_id)]
            if name == "Sin_Tapa":
                cv2.imwrite(SAVE_PATH, annotated)
                print("📸 Imagen guardada (Sin_Tapa)")

                break

        cv2.imshow("iCAM-540 YOLO", annotated)
        
        # -------- FPS --------
        now = time.time()
        if now - last_fps_time >= 1:
            cam_fps = cam_count
            yolo_fps = yolo_count
            cam_count = 0
            yolo_count = 0
            last_fps_time = now
            print(f"🎥 FPS Cámara: {cam_fps} | 🧠 FPS YOLO: {yolo_fps}")

        key  = cv2.waitKey(1) & 0xFF
        if  key == 27:
            break
        elif key == 43 : # '+'
            camera.focus.direction = 0 # lens focusing motor forward
            try:
                camera.focus.distance = 30
                print("lens motor posistion: ", camera.focus.position())
            except ValueError:
                print("lens position out of index")
                    
        elif key == 45: # '-'
            camera.focus.direction = 1 # lens focusing motor backward
            try:
                camera.focus.distance = 30
                print("lens motor posistion: ", camera.focus.position())

            except ValueError:
                print("lens position out of index")
        else:
            continue        
        
        
    # ================= CLEAN =================
    camera.lighting.selector = 0
    cn2.advcam_register_new_image_handler(camera, None)
    cn2.advcam_close(camera)
    cv2.destroyAllWindows()
