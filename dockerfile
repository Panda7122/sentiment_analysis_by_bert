FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/G36maid/sentiment_analysis_by_bert.git

WORKDIR /app/sentiment_analysis_by_bert

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY loadBert.py /app/sentiment_analysis_by_bert
COPY simplebert.py /app/sentiment_analysis_by_bert
COPY TrainOurBert.py /app/sentiment_analysis_by_bert

CMD ["python", "TrainOurBert.py"]
