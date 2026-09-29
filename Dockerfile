# ChessScribe — imagen para despliegue público (sin clave VLM, visión clásica).
# Incluye OpenCV (vía opencv-python-headless) y Stockfish (binario del sistema)
# para que funcionen el reconocimiento por foto y la recomendación de jugada.

FROM python:3.11-slim

# Stockfish y librerías del sistema que necesita OpenCV en tiempo de ejecución.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        stockfish \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Stockfish queda en el PATH; lo exponemos explícitamente por si acaso.
ENV STOCKFISH_PATH=/usr/games/stockfish
ENV ANALYSIS_DEPTH=15
# No se usa clave: el reconocimiento es por visión clásica.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias primero (mejor cacheo de capas).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código.
COPY chesscribe ./chesscribe
COPY backend ./backend
COPY frontend ./frontend
COPY pyproject.toml README.md ./

# La plataforma (Render/Fly/Railway) inyecta el puerto en la variable PORT.
ENV PORT=8000
EXPOSE 8000

# Arranque: uvicorn escuchando en el puerto que indique la plataforma.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
