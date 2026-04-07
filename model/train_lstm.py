import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# 1. LOAD DATA
# =========================

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

print("📦 Data loaded")
print("Train shape:", X_train.shape)

# =========================
# 2. BUILD MODEL
# =========================

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 14)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae']
)

model.summary()

# =========================
# 3. TRAIN MODEL
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# =========================
# 4. EVALUATE
# =========================

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 RMSE: {rmse:.2f}")
print(f"📊 MAE: {mae:.2f}")

# =========================
# 5. CLASSIFICATION LOGIC
# =========================

def classify_rul(rul):
    if rul > 80:
        return "Safe"
    elif rul > 30:
        return "Warning"
    else:
        return "Critical"

# Example predictions
print("\n🔍 Sample Predictions:")
for i in range(5):
    pred = y_pred[i][0]
    print(f"Predicted RUL: {pred:.2f} → {classify_rul(pred)}")

# =========================
# 6. SAVE MODEL
# =========================

model.save("model/lstm_model.h5")
print("\n💾 Model saved as model/lstm_model.h5")

# =========================
# 7. PLOTS
# =========================

# Loss curve
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("model/loss_curve.png")

# Prediction vs Actual
plt.figure()
plt.plot(y_test[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.legend()
plt.title("RUL Prediction vs Actual")
plt.savefig("model/prediction_plot.png")

print("📈 Plots saved in model/")