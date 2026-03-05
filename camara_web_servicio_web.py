import os
import cv2
from flask import Flask, Response
from ultralytics import YOLO

os.system('clear')

app = Flask(__name__)
model = YOLO("best.pt")
cap = cv2.VideoCapture(10)

SAVE_PATH = "/home/icam-540/capturas/captura_sin_tapa.jpg"   # ruta donde guardar


def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        # -----------------------------------------------------
        # Detectar si aparece la clase "Sin_Tapa" en el frame
        # -----------------------------------------------------
        detected_classes = results[0].boxes.cls.tolist()  # IDs detectados

        # Buscar si el nombre "Sin_Tapa" aparece
        for cls_id in detected_classes:
            class_name = results[0].names[int(cls_id)]
            if class_name == "Sin_Tapa":
                cv2.imwrite(SAVE_PATH, annotated)
               
                break

        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
