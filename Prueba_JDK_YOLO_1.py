#!/usr/bin/env python3

import cv2
import numpy
import time
from CamNavi2 import CamNavi2
from ultralytics import YOLO

# --- CONFIGURACIÓN YOLO ---
MODEL_PATH = "best.pt"
SAVE_PATH = "/home/icam-540/capturas/captura_sin_tapa_2.jpg"
model = YOLO(MODEL_PATH)

image_arr = None
icam_color = 0

# ---- Declaracion de varibles para los FPS
fps_count = 0 
fps_start_time = time.time()
fps_value = 0

def gst_to_opencv(sample):
    buf = sample.get_buffer()
    buffer = buf.extract_dup(0, buf.get_size())
    arr = numpy.frombuffer(buffer, dtype=numpy.uint8)
    
    # Decodificar según si la cámara es color o mono
    if icam_color == 1:
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return im

def new_image_handler(sample):
    global image_arr , fps_count, fps_start_time, fps_value
    if sample is None:
        return
    

    fps_count += 1
    elapsed = time.time() - fps_start_time
    if elapsed >=1.0:
        fps_value = fps_count / elapsed
        fps_count = 0
        fps_start_time = time.time()
        print(f"FPS reales camara  {fps_value:.2f}")


    # 1. Convertir el frame del SDK a formato OpenCV
    frame = gst_to_opencv(sample)
    
    # 2. Inferencia YOLO
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # 3. Lógica de validación "Sin_Tapa"
    detected_classes = results[0].boxes.cls.tolist()
    for cls_id in detected_classes:
        class_name = results[0].names[int(cls_id)]
        if class_name == "Sin_Tapa":
            cv2.imwrite(SAVE_PATH, annotated_frame)
            print("¡Detección guardada!")
            break

    # Guardar para mostrar en el bucle principal
    image_arr = annotated_frame

if __name__ == '__main__':
    try: 
        cn2 = CamNavi2.CamNavi2()
    except: 
        cn2 = CamNavi2()
    
    # Configuración de resolución para iCAM-540/500
    width = 1920
    height = 1080

    camera_dict = cn2.enum_camera_list()
    sdk_info = cn2.get_info()
    camera = cn2.get_device_by_name('iCam500') # Ajustar nombre si es iCam540

    icam_color = int(cn2.advcam_query_fw_sku(camera))
    
    # Configurar pipeline
    pipe_params = {'acq_mode':0, 'width':width, 'height':height, 'enable_infer':0}
    if icam_color == 1:
        pipe_params['format'] = 'YUY2'
    
    cn2.advcam_config_pipeline(camera, **pipe_params)
    cn2.advcam_open(camera, -1)

    # Registro del manejador que ahora incluye YOLO
    cn2.advcam_register_new_image_handler(camera, new_image_handler)
    camera.lighting.gain = 100
    camera.lighting.selector = 2
    cn2.advcam_set_img_sharpness(camera,90)
    cn2.advcam_play(camera)
    # Set lens focusing motor to 0
    camera.focus.pos_zero()
    
    camera.focus.direction = 0
    camera.focus.distance = 30
    camera.focus.distance = 30
    time.sleep(2)

    # Ajustes de imagen opcionales
    cn2.advcam_set_img_brightness(camera, 50)
    cn2.advcam_set_img_gain(camera, 5)
    camera.set_acq_frame_rate(30) # Reducido para dar tiempo a la CPU/NPU de procesar YOLO

    print("Presiona ESC para salir...")
    
    while True:
        try:
            if image_arr is not None:
                cv2.imshow("iCAM YOLO Inference", image_arr)
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
            else:
                # Pequeña pausa si no hay imagen para no saturar la CPU
                time.sleep(0.01)
        except KeyboardInterrupt:
            break

    # Limpieza
    camera.lighting.selector = 0
    cn2.advcam_register_new_image_handler(camera, None)    
    cn2.advcam_close(camera)
   
    cv2.destroyAllWindows()