FROM python:apline

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ .

EXPOSE 4000

VOLUME [ "/chroma_langchain_db", "/secrets" ]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]
