# Dockerfile для Streamlit UI
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Кэшировать установки зависимостей
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Копировать приложение
COPY . /app

EXPOSE 8501

# По умолчанию запускать app.py — при необходимости замените имя файла
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]