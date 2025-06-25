# Stage 1: Builder - Bağımlılıkları kurmak ve derlemek için
FROM python:3.13-slim-bullseye AS builder
# Python 3.13 kullandığımız için base imajı da ona göre seçtim.
# "bullseye" Debian'ın güncel stabil sürümüdür. "buster" da olabilirdi.

LABEL maintainer="Bahadir T. <abahadirt@gmail.com>"
LABEL description="YOLO ONNX Microservice for object detection."

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_PREFER_BINARY=1


ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

#    build-essential \ deleted
# Sistem bağımlılıkları (OpenCV-headless ve ONNX runtime için gerekebilir)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/venv_build
RUN python -m venv .venv_app
ENV PATH="/opt/venv_build/.venv_app/bin:$PATH"

# requirements.txt kopyala ve sanal ortama kur
# Bu katmanı kod kopyalamadan önce yapmak, kod değiştiğinde bağımlılıkların tekrar kurulmasını engeller (cache kullanımı)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Stage 2: Runner - Çalıştırma ortamı
FROM python:3.13-slim-bullseye AS runner

ENV PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Güvenlik için root olmayan bir kullanıcı oluştur ve kullan
ARG APP_USER=appuser
ARG APP_GROUP=appgroup
RUN groupadd -r ${APP_GROUP} && useradd -r -g ${APP_GROUP} ${APP_USER}

WORKDIR /app

# Builder aşamasından sadece sanal ortamı ve gerekli sistem kütüphanelerini kopyala
#COPY --from=builder /opt/venv_build/.venv_app /opt/venv_app
COPY --chown=${APP_USER}:${APP_GROUP} --from=builder /opt/venv_build/.venv_app /opt/venv_app



# Gerekli sistem kütüphanelerini runner imajına da kur (ya da base imajda olmalı)
# libgomp1 genellikle python:slim-bullseye'da zaten vardır veya runner'da da kurulabilir.
# Runner'da da sistem bağımlılıklarını kurmak daha güvenli olabilir:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Sanal ortamı PATH'e ekle
ENV PATH="/opt/venv_app/bin:$PATH"

# Uygulama kodunu ve modelleri kopyala
#COPY ./app /app/app
#COPY ./onnx_models /app/onnx_models
COPY --chown=${APP_USER}:${APP_GROUP} ./app /app/app
COPY --chown=${APP_USER}:${APP_GROUP} ./onnx_models /app/onnx_models


# Uygulama kullanıcısına geç
USER ${APP_USER}
# Portu dışarıya aç (FastAPI/Uvicorn varsayılanı)
EXPOSE 8000

# Uygulamayı çalıştır (FastAPI/Uvicorn örneği)
# app/main.py içinde app = FastAPI() varsa:
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["/opt/venv_app/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
