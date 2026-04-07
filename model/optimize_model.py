import tensorflow as tf
import numpy as np
import os
import tensorflow_model_optimization as tfmot
from sklearn.metrics import mean_squared_error

# =========================
# LOAD MODEL + DATA
# =========================

model = tf.keras.models.load_model("model/lstm_model.h5")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# =========================
# ORIGINAL SIZE
# =========================

original_size = os.path.getsize("model/lstm_model.h5") / 1024
print(f"📦 Original model size: {original_size:.2f} KB")

# =========================
# FLOAT32 TFLITE
# =========================

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🔥 LSTM FIX
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("model/lstm_model.tflite", "wb") as f:
    f.write(tflite_model)

float_size = os.path.getsize("model/lstm_model.tflite") / 1024
print(f"📦 TFLite (float32): {float_size:.2f} KB")

# =========================
# INT8 QUANTIZATION
# =========================

def representative_data():
    for i in range(100):
        yield [X_test[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data

# 🔥 LSTM FIX AGAIN
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_int8 = converter.convert()

with open("model/lstm_model_int8.tflite", "wb") as f:
    f.write(tflite_int8)

int8_size = os.path.getsize("model/lstm_model_int8.tflite") / 1024
print(f"📦 INT8 model: {int8_size:.2f} KB")

# =========================
# PRUNING
# =========================

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
        target_sparsity=0.5,
        begin_step=0
    )
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

pruned_model.compile(optimizer='adam', loss='mse')

print("\n✂️ Fine-tuning pruned model...")

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
]

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

pruned_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    verbose=1,
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
)

# Strip pruning
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

# Convert pruned model
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 🔥 LSTM FIX AGAIN
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

pruned_tflite = converter.convert()

with open("model/lstm_model_pruned.tflite", "wb") as f:
    f.write(pruned_tflite)

pruned_size = os.path.getsize("model/lstm_model_pruned.tflite") / 1024
print(f"📦 Pruned model: {pruned_size:.2f} KB")

# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Original RMSE: {rmse:.2f}")

# =========================
# SUMMARY
# =========================

print("\n📊 MODEL COMPARISON")
print(f"Original: {original_size:.2f} KB")
print(f"TFLite: {float_size:.2f} KB")
print(f"INT8: {int8_size:.2f} KB")
print(f"Pruned: {pruned_size:.2f} KB")