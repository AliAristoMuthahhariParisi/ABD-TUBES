# ABD-TUBES

## Sumber Dataset
kaggle: https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

# Set-Up
## 1. Bronze layer
### 1.1 Buat struktur folder proyek
```bash
mkdir -p ~/tubes_k11/data/raw-data
cd ~/tubes_k11
```
### 1.2 Download Dataset
```bash
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

