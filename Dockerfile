FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p artifacts reports logs
EXPOSE 8000
# Models are built inside the image at startup, avoiding cross-version pickle incompatibility.
CMD ["sh", "-c", "python -m src.train --data-dir ${INPUT_DIR:-data/sample} && python -m src.plots --data-dir ${INPUT_DIR:-data/sample} && exec python -m src.app --host 0.0.0.0 --port 8000"]
