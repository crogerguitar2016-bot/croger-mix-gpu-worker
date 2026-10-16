FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN python -m pip install \
    --no-cache-dir \
    --upgrade pip && \
    python -m pip install \
    --no-cache-dir \
    -r requirements.txt

COPY handler.py /app/
COPY test_input.json /app/

CMD ["python", "-u", "/app/handler.py"]
