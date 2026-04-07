import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =========================
# CONFIG
# =========================

DATA_PATH = "data/CMaps/train_FD001.txt"
SEQ_LEN = 30

SELECTED_SENSORS = [
    'sensor_2','sensor_3','sensor_4','sensor_7','sensor_8',
    'sensor_9','sensor_11','sensor_12','sensor_13','sensor_14',
    'sensor_15','sensor_17','sensor_20','sensor_21'
]

# =========================
# 1. LOAD DATA
# =========================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ Put train_FD001.txt inside data/ folder")

print("📂 Loading dataset...")

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)

# Column names
columns = ['unit', 'cycle'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

df.columns = columns

print("📊 Dataset shape:", df.shape)

# =========================
# 2. COMPUTE RUL
# =========================

print("🧠 Computing RUL...")

max_cycles = df.groupby('unit')['cycle'].max()

df['RUL'] = df.apply(
    lambda row: max_cycles[row['unit']] - row['cycle'],
    axis=1
)

df['RUL'] = df['RUL'].clip(upper=125)

print("✅ RUL calculated")

# =========================
# 3. SELECT FEATURES
# =========================

df = df[['unit', 'cycle'] + SELECTED_SENSORS + ['RUL']]

# =========================
# 4. NORMALIZATION
# =========================

print("📏 Normalizing...")

scaler = MinMaxScaler()
df[SELECTED_SENSORS] = scaler.fit_transform(df[SELECTED_SENSORS])

print("✅ Normalization done")

# =========================
# 5. CREATE SEQUENCES
# =========================

print("🔄 Creating sequences...")

def create_sequences(data, seq_len):
    X, y = [], []

    for unit in data['unit'].unique():
        unit_data = data[data['unit'] == unit]

        if len(unit_data) < seq_len:
            continue

        for i in range(len(unit_data) - seq_len):
            X.append(unit_data[SELECTED_SENSORS].iloc[i:i+seq_len].values)
            y.append(unit_data['RUL'].iloc[i+seq_len])

    return np.array(X), np.array(y)

X, y = create_sequences(df, SEQ_LEN)

print("📦 Sequences created")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 6. TRAIN / TEST SPLIT
# =========================

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("✅ Split done")
print("Train:", X_train.shape)
print("Test:", X_test.shape)

# =========================
# 7. SAVE
# =========================

np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("💾 Data saved successfully")

print("\n🎉 PREPROCESSING COMPLETE")