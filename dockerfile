FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY loadBert.py /app
COPY simplebert.py /app
COPY TrainOurBert.py /app

#CMD ["python", "TrainOurBert.py"]
ENTRYPOINT ["python", "TrainOurBert.py"]
