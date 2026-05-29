from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation

# ─────────────────────────────────────────
# 1. INISIALISASI SPARK SESSION
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("tubes_k11_gold") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("✓ Spark Session aktif:", spark.version)

# ─────────────────────────────────────────
# 2. BACA DATA DARI SILVER
# ─────────────────────────────────────────
df_silver = spark.read.parquet("/data/silver/energy_weather_clean")

print(f"✓ Silver dibaca: {df_silver.count()} baris, {len(df_silver.columns)} kolom")

# Registrasi sebagai temp view untuk Spark SQL
df_silver.createOrReplaceTempView("energy_weather")

# ─────────────────────────────────────────
# 3. DESCRIPTIVE ANALYTICS — SPARK SQL
# ─────────────────────────────────────────
print("\n=== DESCRIPTIVE ANALYTICS ===")

# 3.1 Rata-rata konsumsi per jam
print("\n[1] Rata-rata total load actual per jam:")
df_per_jam = spark.sql("""
    SELECT hour,
           ROUND(AVG(`total load actual`), 2)    AS avg_load_actual,
           ROUND(AVG(`total load forecast`), 2)  AS avg_load_forecast,
           ROUND(AVG(`price actual`), 2)          AS avg_price_actual
    FROM energy_weather
    GROUP BY hour
    ORDER BY hour
""")
df_per_jam.show(24, truncate=False)

# 3.2 Rata-rata konsumsi per bulan
print("\n[2] Rata-rata total load actual per bulan:")
df_per_bulan = spark.sql("""
    SELECT month,
           ROUND(AVG(`total load actual`), 2)   AS avg_load_actual,
           ROUND(AVG(`generation solar`), 2)     AS avg_solar,
           ROUND(AVG(`generation wind onshore`), 2) AS avg_wind_onshore,
           ROUND(AVG(`price actual`), 2)          AS avg_price_actual
    FROM energy_weather
    GROUP BY month
    ORDER BY month
""")
df_per_bulan.show(12, truncate=False)

# 3.3 Perbandingan weekend vs weekday
print("\n[3] Rata-rata load: Weekday vs Weekend:")
df_weekend = spark.sql("""
    SELECT is_weekend,
           ROUND(AVG(`total load actual`), 2)  AS avg_load_actual,
           ROUND(AVG(`price actual`), 2)         AS avg_price_actual,
           COUNT(*)                              AS jumlah_jam
    FROM energy_weather
    GROUP BY is_weekend
    ORDER BY is_weekend
""")
df_weekend.show(truncate=False)

# 3.4 Tren produksi energi terbarukan per tahun
print("\n[4] Tren produksi energi terbarukan per tahun:")
df_renewable = spark.sql("""
    SELECT year,
           ROUND(AVG(`generation solar`), 2)           AS avg_solar,
           ROUND(AVG(`generation wind onshore`), 2)    AS avg_wind_onshore,
           ROUND(AVG(`generation hydro run-of-river and poundage`), 2) AS avg_hydro,
           ROUND(AVG(`total load actual`), 2)          AS avg_total_load
    FROM energy_weather
    GROUP BY year
    ORDER BY year
""")
df_renewable.show(truncate=False)

# 3.5 Simpan hasil agregasi ke gold
df_per_jam.write.mode("overwrite").parquet("/data/gold/agg_per_jam")
df_per_bulan.write.mode("overwrite").parquet("/data/gold/agg_per_bulan")
df_renewable.write.mode("overwrite").parquet("/data/gold/agg_renewable")
print("\n✓ Hasil agregasi tersimpan ke /data/gold/")

# ─────────────────────────────────────────
# 4. FEATURE SELECTION — KORELASI PEARSON
# ─────────────────────────────────────────
print("\n=== FEATURE SELECTION — PEARSON CORRELATION ===")

# Kandidat fitur cuaca + waktu yang akan dicek korelasinya
# terhadap target: total load actual
candidate_features = [
    "temp", "pressure", "humidity", "wind_speed",
    "hour", "day_of_week", "month", "is_weekend",
    "forecast solar day ahead", "forecast wind onshore day ahead",
    "total load forecast"
]

# Drop baris yang masih null di kolom kandidat atau target
target_col = "total load actual"
cols_needed = candidate_features + [target_col]
df_corr = df_silver.select(cols_needed).dropna()

# Hitung korelasi Pearson satu per satu terhadap target
print(f"\nKorelasi Pearson terhadap '{target_col}':")
print(f"{'Fitur':<40} {'Korelasi':>10}")
print("-" * 52)

corr_results = []
for feat in candidate_features:
    corr_val = df_corr.stat.corr(feat, target_col)
    corr_results.append((feat, corr_val))
    print(f"{feat:<40} {corr_val:>10.4f}")

# Pilih fitur dengan |korelasi| > 0.1
selected_features = [f for f, c in corr_results if abs(c) > 0.1]
print(f"\n✓ Fitur terpilih (|corr| > 0.1): {selected_features}")

# ─────────────────────────────────────────
# 5. PERSIAPAN DATASET UNTUK MODELING
#    VectorAssembler + StandardScaler
# ─────────────────────────────────────────
print("\n=== PERSIAPAN DATASET MODELING ===")

df_model = df_silver.select(selected_features + [target_col]).dropna()

# VectorAssembler — gabungkan fitur jadi satu vektor
assembler = VectorAssembler(
    inputCols=selected_features,
    outputCol="features_raw"
)
df_assembled = assembler.transform(df_model)

# StandardScaler — normalisasi fitur
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

# Rename target agar tidak ada spasi (lebih aman untuk MLlib)
df_gold_model = df_scaled.select(
    F.col("features"),
    F.col(target_col).alias("label")
)

print(f"✓ Dataset modeling siap: {df_gold_model.count()} baris")
df_gold_model.show(5, truncate=True)

# ─────────────────────────────────────────
# 6. SIMPAN DATASET MODELING KE GOLD
# ─────────────────────────────────────────
df_gold_model.write.mode("overwrite").parquet("/data/gold/dataset_modeling")

print("\n✓ Dataset modeling tersimpan ke /data/gold/dataset_modeling")
print("\n=== Gold Layer Selesai ✓ ===")

spark.stop()
