FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY result.json .

EXPOSE 8501
ENV STREAMLIT_PORT=8501

CMD ["sh", "-c", "streamlit run src/ui/streamlit.py --server.port=${PORT:-$STREAMLIT_PORT} --server.address=0.0.0.0"]

