
import os
import cv2
import numpy as np

import time
import threading
import os
from pathlib import Path
from ultralytics import YOLO
from CamNavi2 import CamNavi2

# ================= CONFIG =================
SAVE_PATH = "/home/icam-540/capturas"
FILE_NAME = "Imagen_trigger.jpg"
MODEL_PATH = "/home/icam-540/Proyectos/Github/Entrenamiento_Deteccion_Objeto/best.pt"
# Resolución cámara (reducida para mejor rendimiento)
WIDTH  = 640   
HEIGHT = 640   
# Tamaño YOLO (pequeño = procesamiento rápido)
YOLO_SIZE_W = 640 
YOLO_SIZE_H = 480 
YOLO_CONF = 0.3  # Confianza mínima (> 0.5 = más rápido)

# FPS objetivo
TARGET_FPS = 40
MUESTRA_IMAGEN = False
detection_event = threading.Event()
# =========================================
model = YOLO(MODEL_PATH)
os.makedirs(SAVE_PATH, exist_ok=True)

image_arr = None
resized_2 = None

lista_archivo = []
def lectura_Confisistema():
    global lista_archivo
    try:
        lista_archivo.clear()
        with open("/home/icam-540/CONFISISTEMA.txt","r", encoding="utf-8") as archivo:
            for linea in archivo:
                linea = linea.replace('\n','')
                print(linea)
                lista_archivo.append(linea)
    except Exception as ex:
        print(f"Error lectura CONFISISTEMA {ex}")


# ---------- Convert GST buffer to OpenCV ----------
def gst_to_opencv(sample):
    buf = sample.get_buffer()
    buffer = buf.extract_dup(0, buf.get_size())
    arr = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ---------- IMAGE CALLBACK (se llama por TRIGGER HARDWARE) ----------
def new_image_handler(sample):
    global image_arr
    global MUESTRA_IMAGEN
    if sample is None:
        return

    img = gst_to_opencv(sample)
    image_arr = img



def save_detection(frame, class_name="Sin_Tapa"):
    """Guarda imagen de detección"""
    try:
        cv2.imwrite(str(SAVE_PATH)+"/"+ str(FILE_NAME), frame)
        print(f"✅ Detección guardada: {SAVE_PATH}")
        detection_event.set()
    except Exception as e:
        print(f"❌ Error al guardar: {e}")


if __name__ == '__main__':
    lectura_Confisistema()

    try:
        cn2 = CamNavi2.CamNavi2()
    except:
        cn2 = CamNavi2()


    # time.sleep(1)

    # for i in range(10):
   #      if os.path.exists("dev/video0"):
     #        print("Video0:")
    #         break
    #     time.sleep(1)
    
    # Enumerar cámaras
    camera_dict = cn2.enum_camera_list()
    print("Cámaras detectadas:", camera_dict)

    camera = cn2.get_device_by_name('iCam500')  # iCAM-540 usa este driver
    icam_color = int(cn2.advcam_query_fw_sku(camera))

    # ---------- PIPELINE ----------
    pipe_params = {
        "acq_mode": 2,
        "width": WIDTH,
        "height": HEIGHT,
        "enable_infer": 0
    }

    if icam_color == 1:
        pipe_params["format"] = "YUY2"

    cn2.advcam_config_pipeline(camera, **pipe_params)
    cn2.advcam_open(camera, -1)
    cn2.advcam_play(camera)
    # Setting do0 parameters
   #  camera.dio.do0.op_mode = 0 # DO op mode: user output
   #  camera.dio.do0.reverse = 0
    camera.dio.do0.user_output = 0 # DO low, DI high
    print("DO lOW " + str(camera.dio.do0.user_output))
   


    # ---------- REGISTER CALLBACK ----------
    cn2.advcam_register_new_image_handler(camera, new_image_handler)


    camera.hw_trigger_delay = 0
    print("Delay " + str(camera.hw_trigger_delay))


    #camera.hw_trigger_delay = 1000
    print("Delay " + str(camera.hw_trigger_delay))
    camera.lighting.selector = int(lista_archivo[0])
    #camera.lighting.selector = 2
    
    #camera.lighting.gain = 50
    camera.lighting.gain = int(lista_archivo[1])

    #cn2.advcam_set_img_sharpness(camera, 5)
    #cn2.advcam_set_img_brightness(camera, 250)
    #cn2.advcam_set_img_gain(camera, 6)
    camera.image.saturation = int(lista_archivo[2])
    camera.image.gamma = int(lista_archivo[3])

    cn2.advcam_set_img_sharpness(camera, int(lista_archivo[4]))
    cn2.advcam_set_img_brightness(camera,  int(lista_archivo[5]))
    cn2.advcam_set_img_gain(camera, int(lista_archivo[6]))

    camera.focus.pos_zero()

    #camera.focus.distance = 65
    camera.focus.distance = int(lista_archivo[7])
    i = 0
    while i < 7:
            camera.focus.direction = 1 # lens focusing motor backward
            try:
                camera.focus.distance = 100
                print("lens motor posistion: ", camera.focus.position())
                time.sleep(0.1) 
                i+=1
                print("valor ", i)
            except ValueError:
                print("lens position out of index")
     #camera.focus.distance = 10
     #camera.focus.direction = 1
    
     #print("lens motor posistion: ", camera.focus.position())
    # ---------- START STREAM ----------
    cn2.advcam_play(camera)

    print("✅ iCAM-540 listo. Esperando trigger hardware en PIN 10...")
    bandera = False
    count = 0
    try:
        while True:
            if image_arr  is not None:
                count+=1
                if count == 2:
                    camera.dio.do0.user_output = 0
                    salida  = str(camera.dio.do0.user_output)
                    print("DO Low " + salida)

                #cv2.imshow("Vista Camara",image_arr) 
                resized = cv2.resize(image_arr, (YOLO_SIZE_W, YOLO_SIZE_H))

                if resized_2 is not None:
                    if np.mean(cv2.absdiff(resized,resized_2)) > 10:
                        bandera = False

                

                results = model(
                    resized, 
                    verbose=False,
                     conf=YOLO_CONF  # Confianza mínima = más rápido
                )

                frame_yolo = results[0].plot()

                if bandera == False:
                    for cls_id in results[0].boxes.cls.tolist():
                        class_name = results[0].names[int(cls_id)]
                        if class_name == "Sin_Tapa":
                            print(f"🎯 Objeto detectado: {class_name}")
                            save_detection(frame_yolo, class_name)
                            resized_2 = resized 
                            bandera = True
                            count = 0
                            camera.dio.do0.user_output = 1
                            salida  = str(camera.dio.do0.user_output)
                            print("DO high " + salida)
                            break

                        if class_name == "Con_Tapa":
                            print(f"🎯 Objeto detectado: {class_name}")
                            resized_2 = resized 
                            bandera = True

                cv2.imshow("Vista Camara",frame_yolo)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break
                elif key == ord('-'): 
                    cv2.destroyAllWindows()
                    image_arr = None
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

            else:
                time.sleep(0.01)  
    except KeyboardInterrupt:
        pass

    # ---------- CLEANUP ----------
    cv2.destroyAllWindows()
    cn2.advcam_register_new_image_handler(camera, None)
    cn2.advcam_close(camera)
