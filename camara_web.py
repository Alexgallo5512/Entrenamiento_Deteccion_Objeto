import os 
import cv2
from ultralytics import YOLO
os.system('clear')



SAVE_PATH = "/home/icam-540/captura/captura_sin_tapa_2.jpg"   # ruta donde guardar


def main():
    
    model = YOLO("best.pt")
    cap=cv2.VideoCapture(0)
    while cap.isOpened():
        # Leemos el frame del video
        ret, frame = cap.read()
        if not ret:
            break
        # Realizamos la inferencia de YOLO sobre el frame
        results = model(frame)
        # Extraemos los resultados
        annotated_frame = results[0].plot()
        #print(annotated_frame)
        # Visualizamos los resultados

        detected_classes = results[0].boxes.cls.tolist()  # IDs detectados

        # Buscar si el nombre "Sin_Tapa" aparece
        for cls_id in detected_classes:
            class_name = results[0].names[int(cls_id)]
            if class_name == "Sin_Tapa":
                cv2.imwrite(SAVE_PATH, annotated_frame)
               
                break

        cv2.imshow("YOLO Inference", annotated_frame)
        # El ciclo se rompe al presionar "Esc"
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release() 
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()