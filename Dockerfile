FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/config.json ./src/
COPY src/ ./src/
COPY models/ ./models/
ENV PYTHONPATH=/app/src
CMD ["python", "src/predict.py"]