FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#COPY loadBert.py /app
COPY token.json /app
COPY predict_emotion.py /app/loadBert.py
#COPY simplebert.py /app
#COPY TrainOurBert.py /app
COPY bot.py /app
COPY model /app/model 

#RUN python simplebert.py

#CMD ["python", "TrainOurBert.py"]

ENTRYPOINT ["python", "bot.py"]
