FROM python:3.11

COPY . /app

WORKDIR /app
ENV AM_I_IN_A_DOCKER_CONTAINER 1

RUN pip install -r requirements.txt

CMD ["python3", "camara_web.py"]

