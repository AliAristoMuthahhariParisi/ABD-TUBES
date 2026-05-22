# ABD-TUBES

## Sumber Dataset
kaggle: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

# Set-Up
Pastikan ada docker dan wsl

## 1. Bronze layer
### 1.1 Buat struktur folder proyek
```bash
mkdir -p ~/tubes_k11/data/raw-data
cd ~/tubes_k11
```
### 1.2 Download Dataset
```bash
cd ~/tubes_k11/data/raw-data

# Unduh energy_dataset
wget https://raw.githubusercontent.com/AliAristoMuthahhariParisi/ABD-TUBES/refs/heads/main/Dataset/energy_dataset.csv

# unduh weather_features
wget https://raw.githubusercontent.com/AliAristoMuthahhariParisi/ABD-TUBES/refs/heads/main/Dataset/weather_features.csv
```
### 1.3 Buat docker-compose.yml
``` bash
cd ~/tubes_k11
nano docker-compose.yml
```
Isi:
```bash
version: '3.8'
services:
  minio:
    image: minio/minio:latest
    container_name: tubes-k11-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: admin123
    volumes:
      - ./data:/data
    command: server /data --console-address ":9001"
```
Simpan: Ctrl+X → Y → Enter

### 1.4 Jalankan MinIO
```bash
cd ~/tubes_k11
docker compose up -d

# Cek container berjalan
docker ps
```
Buka browser di Windows: 👉 http://localhost:9001
Login: admin / admin123

Buat 3 bucket: bronze, silver, gold

### 1.5 Upload dataset ke bucket Bronze
```bash
# Install MinIO client (mc)
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

# Hubungkan mc ke MinIO lokal
mc alias set local http://localhost:9000 admin admin123

# Upload kedua CSV ke bucket bronze
mc cp ~/tubes_k11/data/raw-data/energy_dataset.csv local/bronze/
mc cp ~/tubes_k11/data/raw-data/weather_features.csv local/bronze/

# Verifikasi
mc ls local/bronze/
```
# Set-Up Spark
```bash
cd ~/tubes_k11
wget https://archive.apache.org/dist/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz

# Buat DockerFile
```
Isi: 
```bash
FROM python:3.11-slim

# Install Java (wajib untuk Spark)
RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Copy dan ekstrak Spark
COPY spark-3.5.5-bin-hadoop3.tgz /opt/
RUN tar -xzf /opt/spark-3.5.5-bin-hadoop3.tgz -C /opt/ \
    && mv /opt/spark-3.5.5-bin-hadoop3 /opt/spark \
    && rm /opt/spark-3.5.5-bin-hadoop3.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYSPARK_PYTHON=python3

# Install library Python
RUN pip install --no-cache-dir \
    pyspark==3.5.5 \
    pandas \
    matplotlib \
    seaborn \
    boto3

WORKDIR /app
```
Simpan: Ctrl+X → Y → Enter

## 2. Silver Layer

Silver layer digunakan untuk membersihkan data mentah dari Bronze, mengubah tipe data timestamp, menangani missing values, membuat fitur waktu, mengagregasi data cuaca dari beberapa kota, lalu menggabungkan dataset energi dan cuaca menjadi data bersih siap analisis.


