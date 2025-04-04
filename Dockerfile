FROM python:3.11-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt tensorflow

CMD ["python3", "app.py"]