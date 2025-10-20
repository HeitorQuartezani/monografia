# Imagem base com Python (Debian Bullseye para compatibilidade)
FROM python:3.9-slim-bullseye

# 1. Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    wget \
    firefox-esr \
    xvfb \
    libssl1.1 \
    libcurl4 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 2. Instala MongoDB manualmente
ENV MONGO_VERSION 6.0.11
RUN wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-debian11-${MONGO_VERSION}.tgz \
 && tar -xvzf mongodb-linux-x86_64-debian11-${MONGO_VERSION}.tgz \
 && mv mongodb-linux-x86_64-debian11-${MONGO_VERSION}/bin/* /usr/local/bin/ \
 && rm -rf mongodb-*

# 3. Configura Geckodriver para Selenium
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.30.0/geckodriver-v0.30.0-linux64.tar.gz \
 && tar -xvzf geckodriver* \
 && chmod +x geckodriver \
 && mv geckodriver /usr/local/bin/

# 4. Configuração do MongoDB
RUN mkdir -p /data/db \
 && chown -R `whoami` /data/db

# 5. Diretório de trabalho
WORKDIR /app
COPY . .

# 6. Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# 7. Script de inicialização
CMD mongod --fork --logpath /var/log/mongodb.log && \
    sleep 5 && \
    python data_collection.py && \
    python text_processing.py && \
    python embedding_generation.py