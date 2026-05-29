```bash

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ─────────────────────────────────────────
# 1. INISIALISASI SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("tubes_k11_silver") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("✓ Spark Session aktif:", spark.version)

# ─────────────────────────────────────────
# 2. BACA DATA MENTAH DARI BRONZE (CSV)
# ─────────────────────────────────────────
df_energy = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("/data/raw-data/energy_dataset.csv")

df_weather = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv("/data/raw-data/weather_features.csv")

print(f"✓ energy_dataset  : {df_energy.count()} baris, {len(df_energy.columns)} kolom")
print(f"✓ weather_features: {df_weather.count()} baris, {len(df_weather.columns)} kolom")

# ─────────────────────────────────────────
# 3. CLEANING — FORWARD FILL MISSING VALUES
# energy pakai kolom "time", weather pakai "dt_iso"
# ─────────────────────────────────────────

# -- energy_dataset --
window_energy = Window.orderBy("time").rowsBetween(Window.unboundedPreceding, 0)
numeric_energy = [c for c, t in df_energy.dtypes if t in ("double", "float", "int", "bigint")]

df_energy_clean = df_energy
for c in numeric_energy:
    df_energy_clean = df_energy_clean.withColumn(
        c, F.last(F.col(c), ignorenulls=True).over(window_energy)
    )

# -- weather_features (orderBy dt_iso bukan time) --
window_weather = Window.orderBy("dt_iso").rowsBetween(Window.unboundedPreceding, 0)
numeric_weather = [c for c, t in df_weather.dtypes if t in ("double", "float", "int", "bigint")]

df_weather_clean = df_weather
for c in numeric_weather:
    df_weather_clean = df_weather_clean.withColumn(
        c, F.last(F.col(c), ignorenulls=True).over(window_weather)
    )

print("✓ Forward-fill selesai")

# ─────────────────────────────────────────
# 4. AGREGASI CUACA 5 KOTA → RATA-RATA NASIONAL
# ─────────────────────────────────────────
weather_num_cols = [c for c, t in df_weather_clean.dtypes
                    if t in ("double", "float", "int", "bigint")]

df_weather_agg = df_weather_clean \
    .groupBy("dt_iso") \
    .agg(*[F.avg(c).alias(c) for c in weather_num_cols])

# Samakan nama kolom waktu agar bisa di-join dengan energy
df_weather_agg = df_weather_agg.withColumnRenamed("dt_iso", "time")

print(f"✓ Weather setelah agregasi: {df_weather_agg.count()} baris")

# ─────────────────────────────────────────
# 5. JOIN ENERGY + WEATHER BY TIMESTAMP
# ─────────────────────────────────────────
df_joined = df_energy_clean.join(
    df_weather_agg,
    on="time",
    how="left"
)

print(f"✓ Setelah join: {df_joined.count()} baris, {len(df_joined.columns)} kolom")

# ─────────────────────────────────────────
# 6. EKSTRAKSI FITUR WAKTU
# ─────────────────────────────────────────
df_silver = df_joined \
    .withColumn("timestamp",   F.to_timestamp(F.col("time"))) \
    .withColumn("hour",        F.hour("timestamp")) \
    .withColumn("day_of_week", F.dayofweek("timestamp")) \
    .withColumn("month",       F.month("timestamp")) \
    .withColumn("year",        F.year("timestamp")) \
    .withColumn("is_weekend",  (F.dayofweek("timestamp").isin([1, 7])).cast("int"))

print("✓ Fitur waktu diekstraksi: hour, day_of_week, month, year, is_weekend")

# ─────────────────────────────────────────
# 7. SIMPAN KE SILVER (PARQUET)
# ─────────────────────────────────────────
### Konversi dataset awal ke format Parquet untuk optimasi pemrosesan

output_path = "/data/silver/energy_weather_clean"

df_silver.write \
    .mode("overwrite") \
    .parquet(output_path)

print(f"✓ Silver tersimpan ke   : {output_path}")
print(f"✓ Total baris silver    : {df_silver.count()}")
print(f"✓ Total kolom silver    : {len(df_silver.columns)}")
print("\n=== Silver Layer Selesai ✓ ===")

spark.stop()

```
