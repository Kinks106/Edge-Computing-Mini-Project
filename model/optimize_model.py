import tensorflow as tf
import numpy as np
import os
import tensorflow_model_optimization as tfmot
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CREATE OUTPUT FOLDERS
# =========================

os.makedirs("model", exist_ok=True)
os.makedirs("model/optimized_preds", exist_ok=True)

# =========================
# LOAD MODEL + DATA
# =========================

model = tf.keras.models.load_model("model/lstm_model.h5")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

# =========================
# ORIGINAL SIZE
# =========================

original_size = os.path.getsize("model/lstm_model.h5") / 1024
print(f"📦 Original model size: {original_size:.2f} KB")

# =========================
# FLOAT32 TFLITE
# =========================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
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

pruned_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    verbose=1,
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
)

pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

# Convert pruned model
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
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
y_pred_pruned = pruned_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_pruned = np.sqrt(mean_squared_error(y_test, y_pred_pruned))

print(f"\n📊 Original RMSE: {rmse:.2f}")
print(f"📊 Pruned RMSE: {rmse_pruned:.2f}")

# Save predictions
np.save("model/optimized_preds/original_preds.npy", y_pred)
np.save("model/optimized_preds/pruned_preds.npy", y_pred_pruned)

# =========================
# CLASSIFICATION METRICS
# =========================

def classify(rul):
    if rul > 80:
        return 0
    elif rul > 30:
        return 1
    else:
        return 2

y_true_cls = np.array([classify(y) for y in y_test])
y_pred_cls = np.array([classify(y[0]) for y in y_pred])

accuracy = accuracy_score(y_true_cls, y_pred_cls)
precision = precision_score(y_true_cls, y_pred_cls, average='weighted')
recall = recall_score(y_true_cls, y_pred_cls, average='weighted')
f1 = f1_score(y_true_cls, y_pred_cls, average='weighted')

print(f"\n📊 Accuracy: {accuracy:.2f}")
print(f"📊 Precision: {precision:.2f}")
print(f"📊 Recall: {recall:.2f}")
print(f"📊 F1 Score: {f1:.2f}")

# =========================
# PLOTS
# =========================

# Size comparison
plt.figure()
models = ["Original", "TFLite", "INT8", "Pruned"]
sizes = [original_size, float_size, int8_size, pruned_size]
plt.bar(models, sizes)
plt.title("Model Size Comparison")
for i, v in enumerate(sizes):
    plt.text(i, v + 5, f"{v:.1f}", ha='center')
plt.savefig("model/model_size_comparison.png")

# RMSE comparison
plt.figure()
plt.bar(["Original", "Pruned"], [rmse, rmse_pruned])
plt.title("RMSE Comparison")
plt.savefig("model/rmse_comparison.png")

# Confusion Matrix
cm = confusion_matrix(y_true_cls, y_pred_cls)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Safe","Warning","Critical"],
            yticklabels=["Safe","Warning","Critical"])
plt.title("Confusion Matrix")
plt.savefig("model/confusion_matrix.png")

# Classification metrics
plt.figure()
metrics = ["Accuracy", "Precision", "Recall", "F1"]
values = [accuracy, precision, recall, f1]
plt.bar(metrics, values)
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
plt.title("Classification Metrics")
plt.savefig("model/classification_metrics.png")

plt.show()