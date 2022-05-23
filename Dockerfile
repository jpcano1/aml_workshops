FROM python:3.9.7-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /workshops

COPY requirements/requirements.txt .

RUN apt-get update && \
    apt-get install -y libpq-dev build-essential python3-opencv && \
    pip install -r requirements.txt \
    mkdir app

ADD app ./app

CMD gunicorn --bind=0.0.0.0:$PORT --log-level=DEBUG -w=4 --timeout=1080 'app:app'