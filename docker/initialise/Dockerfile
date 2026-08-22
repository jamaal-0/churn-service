FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ml/
COPY data/ data/
COPY env_config/ env_config/
COPY api/scripts/initialise_mlflow.py .

CMD ["python", "initialise_mlflow.py"]