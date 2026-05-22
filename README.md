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
## 2. Silver Layer

Silver layer digunakan untuk membersihkan data mentah dari Bronze, mengubah tipe data timestamp, menangani missing values, membuat fitur waktu, mengagregasi data cuaca dari beberapa kota, lalu menggabungkan dataset energi dan cuaca menjadi data bersih siap analisis.

### 2.1 Tambahkan service Spark Notebook di docker-compose.yml

Buka kembali file `docker-compose.yml`:

```bash
cd ~/tubes_k11
nano docker-compose.yml
```

Ubah isi file menjadi seperti berikut:

```yaml
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

  spark:
    image: jupyter/pyspark-notebook:latest
    container_name: tubes-k11-spark
    ports:
      - "8888:8888"
      - "4040:4040"
    volumes:
      - ./data:/home/jovyan/work/data
      - ./scripts:/home/jovyan/work/scripts
    environment:
      JUPYTER_ENABLE_LAB: "yes"
    command: start-notebook.sh --NotebookApp.token=''
```

Simpan file dengan cara:

```bash
Ctrl + X
Y
Enter
```

Jalankan ulang container:

```bash
docker compose down
docker compose up -d
docker ps
```

Buka Jupyter Notebook di browser:

```bash
http://localhost:8888
```

---

### 2.2 Buat folder untuk script dan output Silver

```bash
cd ~/tubes_k11
mkdir -p scripts
mkdir -p data/silver
```

---

### 2.3 Buat script transformasi Silver

```bash
nano scripts/01_silver_processing.py
```

Isi script berikut:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, hour, dayofweek, month, year,
    avg, when
)
from pyspark.sql.window import Window
from pyspark.sql.functions import last

spark = SparkSession.builder \
    .appName("Silver Layer Processing - Energy and Weather Dataset") \
    .getOrCreate()

# Path input Bronze
energy_path = "/home/jovyan/work/data/raw-data/energy_dataset.csv"
weather_path = "/home/jovyan/work/data/raw-data/weather_features.csv"

# Path output Silver
silver_path = "/home/jovyan/work/data/silver/energy_weather_cleaned"

# Load dataset Bronze
energy_df = spark.read.csv(energy_path, header=True, inferSchema=True)
weather_df = spark.read.csv(weather_path, header=True, inferSchema=True)

print("Schema energy dataset:")
energy_df.printSchema()

print("Schema weather dataset:")
weather_df.printSchema()

# 1. Cleaning energy dataset

energy_df = energy_df.withColumn(
    "time",
    to_timestamp(col("time"), "yyyy-MM-dd HH:mm:ssXXX")
)

# Ambil kolom yang relevan
energy_selected = energy_df.select(
    "time",
    "generation biomass",
    "generation fossil brown coal/lignite",
    "generation fossil gas",
    "generation fossil hard coal",
    "generation hydro pumped storage consumption",
    "generation hydro run-of-river and poundage",
    "generation hydro water reservoir",
    "generation nuclear",
    "generation oil",
    "generation other",
    "generation other renewable",
    "generation solar",
    "generation waste",
    "generation wind onshore",
    "forecast solar day ahead",
    "forecast wind onshore day ahead",
    "total load forecast",
    "total load actual",
    "price day ahead",
    "price actual"
)

# Forward fill untuk missing value berbasis waktu
window_spec = Window.orderBy("time").rowsBetween(Window.unboundedPreceding, 0)

for column_name in energy_selected.columns:
    if column_name != "time":
        energy_selected = energy_selected.withColumn(
            column_name,
            last(col(column_name), ignorenulls=True).over(window_spec)
        )

# Tambahkan fitur waktu
energy_selected = energy_selected.withColumn("hour", hour(col("time"))) \
    .withColumn("day_of_week", dayofweek(col("time"))) \
    .withColumn("month", month(col("time"))) \
    .withColumn("year", year(col("time"))) \
    .withColumn("is_weekend", when(col("day_of_week").isin(1, 7), 1).otherwise(0))

# 2. Cleaning weather dataset

weather_df = weather_df.withColumn(
    "dt_iso",
    to_timestamp(col("dt_iso"), "yyyy-MM-dd HH:mm:ssXXX")
)

weather_selected = weather_df.select(
    col("dt_iso").alias("time"),
    "city_name",
    "temp",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_deg",
    "rain_1h",
    "rain_3h",
    "snow_3h",
    "clouds_all"
)

# Isi nilai null dengan 0 untuk kolom hujan/salju
weather_selected = weather_selected.fillna({
    "rain_1h": 0,
    "rain_3h": 0,
    "snow_3h": 0
})

# Agregasi cuaca 5 kota menjadi rata-rata nasional per timestamp
weather_agg = weather_selected.groupBy("time").agg(
    avg("temp").alias("avg_temp"),
    avg("pressure").alias("avg_pressure"),
    avg("humidity").alias("avg_humidity"),
    avg("wind_speed").alias("avg_wind_speed"),
    avg("wind_deg").alias("avg_wind_deg"),
    avg("rain_1h").alias("avg_rain_1h"),
    avg("rain_3h").alias("avg_rain_3h"),
    avg("snow_3h").alias("avg_snow_3h"),
    avg("clouds_all").alias("avg_clouds_all")
)

# 3. Join energy dan weather

silver_df = energy_selected.join(weather_agg, on="time", how="inner")

# Drop data yang masih null setelah join
silver_df = silver_df.dropna()

print("Jumlah baris Silver:", silver_df.count())
print("Jumlah kolom Silver:", len(silver_df.columns))

silver_df.printSchema()
silver_df.show(5, truncate=False)

# Simpan ke format Parquet
silver_df.write.mode("overwrite").parquet(silver_path)

print("Silver layer berhasil dibuat di:", silver_path)

spark.stop()
```

Simpan file:

```bash
Ctrl + X
Y
Enter
```

---

### 2.4 Jalankan script Silver di container Spark

```bash
docker exec -it tubes-k11-spark bash
cd /home/jovyan/work
python scripts/01_silver_processing.py
exit
```

---

### 2.5 Upload hasil Silver ke bucket MinIO

```bash
mc cp --recursive ~/tubes_k11/data/silver/energy_weather_cleaned local/silver/
mc ls local/silver/
```

Output yang diharapkan adalah folder Parquet hasil proses Silver berhasil masuk ke bucket `silver`.

---

## 3. Gold Layer

Gold layer digunakan untuk membentuk data siap analisis dan data siap model. Pada tahap ini, data Silver diolah menjadi beberapa tabel agregat, yaitu tren konsumsi bulanan, konsumsi per jam, produksi energi terbarukan, serta dataset final untuk modeling prediksi beban energi.

### 3.1 Buat folder output Gold

```bash
cd ~/tubes_k11
mkdir -p data/gold
```

---

### 3.2 Buat script Gold Layer

```bash
nano scripts/02_gold_processing.py
```

Isi script berikut:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, round
)

spark = SparkSession.builder \
    .appName("Gold Layer Processing - Energy Analytics") \
    .getOrCreate()

# Path input Silver
silver_path = "/home/jovyan/work/data/silver/energy_weather_cleaned"

# Path output Gold
gold_base_path = "/home/jovyan/work/data/gold"

# Load data Silver
silver_df = spark.read.parquet(silver_path)

print("Data Silver:")
silver_df.show(5)
print("Jumlah data Silver:", silver_df.count())

# 1. Gold: Tren konsumsi bulanan

monthly_consumption = silver_df.groupBy("year", "month").agg(
    round(avg("total load actual"), 2).alias("avg_total_load_actual"),
    round(avg("total load forecast"), 2).alias("avg_total_load_forecast"),
    round(avg("price actual"), 2).alias("avg_price_actual"),
    round(avg("price day ahead"), 2).alias("avg_price_day_ahead")
).orderBy("year", "month")

monthly_consumption.write.mode("overwrite").parquet(
    f"{gold_base_path}/monthly_consumption"
)

print("Gold monthly_consumption:")
monthly_consumption.show(10)

# 2. Gold: Pola konsumsi per jam

hourly_consumption = silver_df.groupBy("hour").agg(
    round(avg("total load actual"), 2).alias("avg_total_load_actual"),
    round(avg("price actual"), 2).alias("avg_price_actual"),
    round(avg("avg_temp"), 2).alias("avg_temperature")
).orderBy("hour")

hourly_consumption.write.mode("overwrite").parquet(
    f"{gold_base_path}/hourly_consumption"
)

print("Gold hourly_consumption:")
hourly_consumption.show(24)

# 3. Gold: Produksi energi terbarukan

renewable_energy = silver_df.withColumn(
    "total_renewable_generation",
    col("generation solar") +
    col("generation wind onshore") +
    col("generation hydro run-of-river and poundage") +
    col("generation hydro water reservoir") +
    col("generation other renewable")
)

monthly_renewable = renewable_energy.groupBy("year", "month").agg(
    round(avg("generation solar"), 2).alias("avg_solar_generation"),
    round(avg("generation wind onshore"), 2).alias("avg_wind_generation"),
    round(avg("generation hydro run-of-river and poundage"), 2).alias("avg_hydro_generation"),
    round(avg("total_renewable_generation"), 2).alias("avg_total_renewable_generation")
).orderBy("year", "month")

monthly_renewable.write.mode("overwrite").parquet(
    f"{gold_base_path}/monthly_renewable_generation"
)

print("Gold monthly_renewable_generation:")
monthly_renewable.show(10)

# 4. Gold: Dataset siap modeling

modeling_dataset = renewable_energy.select(
    "time",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "avg_temp",
    "avg_pressure",
    "avg_humidity",
    "avg_wind_speed",
    "avg_clouds_all",
    "generation solar",
    "generation wind onshore",
    "generation fossil gas",
    "generation nuclear",
    "price day ahead",
    "price actual",
    "total_renewable_generation",
    col("total load actual").alias("label")
).dropna()

modeling_dataset.write.mode("overwrite").parquet(
    f"{gold_base_path}/modeling_dataset"
)

print("Gold modeling_dataset:")
modeling_dataset.show(5)
print("Jumlah data modeling:", modeling_dataset.count())

spark.stop()
```

Simpan file:

```bash
Ctrl + X
Y
Enter
```

---

### 3.3 Jalankan script Gold

```bash
docker exec -it tubes-k11-spark bash
cd /home/jovyan/work
python scripts/02_gold_processing.py
exit
```

---

### 3.4 Upload hasil Gold ke bucket MinIO

```bash
mc cp --recursive ~/tubes_k11/data/gold/monthly_consumption local/gold/
mc cp --recursive ~/tubes_k11/data/gold/hourly_consumption local/gold/
mc cp --recursive ~/tubes_k11/data/gold/monthly_renewable_generation local/gold/
mc cp --recursive ~/tubes_k11/data/gold/modeling_dataset local/gold/

mc ls local/gold/
```

```

---

## 4. Verifikasi Struktur Medallion

Setelah Bronze, Silver, dan Gold selesai dibuat, struktur data pada MinIO harus berisi:

```bash
bronze/
 ├── energy_dataset.csv
 └── weather_features.csv

silver/
 └── energy_weather_cleaned/

gold/
 ├── monthly_consumption/
 ├── hourly_consumption/
 ├── monthly_renewable_generation/
 └── modeling_dataset/
```

Cek melalui terminal:

```bash
mc ls local/bronze/
mc ls local/silver/
mc ls local/gold/
```

Cek juga melalui browser:

```bash
http://localhost:9001
```

Login:

```bash
Username: admin
Password: admin123
```

---

## 5. Deskripsi Singkat Pipeline

Pipeline ini menggunakan arsitektur Medallion yang terdiri dari tiga lapisan utama. Bronze layer menyimpan data mentah dari Kaggle dalam bentuk CSV tanpa transformasi. Silver layer membersihkan data, mengubah timestamp, menangani missing values, membuat fitur waktu, mengagregasi data cuaca dari lima kota, dan menggabungkannya dengan data energi. Gold layer menghasilkan tabel analitik siap pakai, seperti tren konsumsi bulanan, pola konsumsi per jam, produksi energi terbarukan, serta dataset final untuk modeling prediksi beban energi.

Dengan pipeline ini, data mentah dapat diproses secara lebih terstruktur menggunakan Apache Spark dan disimpan dalam format Parquet agar lebih efisien untuk analisis lanjutan, visualisasi, serta pengembangan model prediksi beban energi.