# 使用 Python 3.8 作為基本映像
FROM python:3.8

# 安裝 git，以便 clone GitHub repo
RUN apt-get update && apt-get install -y git

# 克隆 GitHub repo
RUN git clone https://github.com/G36maid/sentiment_analysis_by_bert.git

# 設置工作目錄
WORKDIR /app/sentiment_analysis_by_bert

# 安裝所需的 Python 套件
RUN pip install -r requirements.txt  
# 假設有一個 requirements.txt 檔案包含所需的套件

# 複製程式檔案到工作目錄
COPY loadBert.py /app/loadBert.py
COPY simplebert.py /app/simplebert.py
COPY TrainOurBert.py /app/TrainOurBert.py

# 運行程式
CMD ["python", "loadBert.py"]
