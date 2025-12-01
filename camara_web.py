import os 
import cv2
import logging
from ultralytics import YOLO
os.system('cls')
logger = logging.getLogger(__name__)
AM_I_IN_A_DOCKER_CONTAINER: bool = os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False)


def main():
    logger.info(AM_I_IN_A_DOCKER_CONTAINER)
    model = YOLO("best.pt")
    cap=cv2.VideoCapture(1)
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
        cv2.imshow("YOLO Inference", annotated_frame)
        # El ciclo se rompe al presionar "Esc"
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()