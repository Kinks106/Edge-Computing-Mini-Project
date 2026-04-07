# 🔧 Edge-Based Predictive Maintenance using LSTM (ESP32 + Wokwi)

## 📌 Overview

This project demonstrates an **Edge-Based Predictive Maintenance System** using a trained LSTM model on industrial sensor data. The system predicts machine health in real-time and classifies it into:

* 🟢 Safe
* 🟡 Warning
* 🔴 Critical

The model is trained offline using NASA’s turbofan engine dataset and deployed in a **lightweight form on an ESP32 (simulated using Wokwi)**.

---

## 🎯 Objectives

* Predict Remaining Useful Life (RUL) using time-series sensor data
* Optimize deep learning model for edge deployment
* Simulate real-time inference using ESP32
* Trigger alerts based on predicted machine condition

---

## 🏗️ System Architecture

```
Sensor Data → LSTM Model (Trained in Python) → Model Optimization → Edge Deployment (ESP32)
```

* **Training (Cloud/PC):** Python + TensorFlow
* **Deployment (Edge):** ESP32 (Wokwi Simulation)
* **Sensors:** Temperature (DHT22), Vibration (Potentiometer)

---

## 📊 Dataset

* **Source:** NASA CMAPSS Turbofan Engine Dataset
* **File Used:** `train_FD001.txt`
* **Features:** 21 sensors + operational settings
* **Selected Sensors:** 14 important sensors
* **Sequence Length:** 30 timesteps
* **Target:** Remaining Useful Life (RUL), capped at 125

---

## 🧠 Model Details

### Architecture:

```
LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)
```

### Training:

* Loss: Mean Squared Error (MSE)
* Optimizer: Adam
* Metrics: RMSE, MAE
* Epochs: 50 (Early Stopping enabled)

### Performance:

* RMSE ≈ 14.35
* MAE ≈ 11.27

---

## ⚙️ Model Optimization

To enable deployment on resource-constrained devices:

| Technique          | Result  |
| ------------------ | ------- |
| Original Model     | 426 KB  |
| TFLite (float32)   | 145 KB  |
| INT8 Quantization  | 52.7 KB |
| Pruned + Quantized | 50.6 KB |

📉 Achieved ~8x reduction in model size

---

## 📡 Edge Simulation (ESP32)

Simulated using **Wokwi**

### Inputs:

* 🌡️ Temperature (DHT22)
* 🎛️ Vibration (Potentiometer)

### Outputs:

* 🟢 Green LED → Safe
* 🟡 Yellow LED → Warning
* 🔴 Red LED → Critical

### Logic:

A lightweight approximation of the trained model is used for real-time inference.

---

## 📁 Project Structure

```
edge_lstm/
│
├── data/
│   ├── train_FD001.txt
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
│
├── model/
│   ├── lstm_model.h5
│   ├── lstm_model.tflite
│   ├── lstm_model_int8.tflite
│   ├── lstm_model_pruned.tflite
│   ├── loss_curve.png
│   └── prediction_plot.png
│
├── wokwi/
│   ├── diagram.json
│   ├── sketch.ino
│   └── libraries.txt
│
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Train Model

```bash
python model/train_lstm.py
```

### 2️⃣ Optimize Model

```bash
python model/optimize_model.py
```

### 3️⃣ Run Edge Simulation

* Open https://wokwi.com/
* Create ESP32 Arduino project
* Paste:

  * `diagram.json`
  * `sketch.ino`
  * `libraries.txt`
* Click ▶ Run

---

## 🧪 Demo

* Adjust temperature and vibration
* Observe LED changes based on predicted state
* Monitor serial output for real-time predictions

---

## 🧠 Key Insight

This project highlights the **trade-off between accuracy and efficiency**:

* High accuracy model → Cloud
* Optimized lightweight model → Edge device

---

## 🚀 Future Work

* Deploy using TensorFlow Lite Micro on real ESP32
* Integrate real industrial sensors
* Use real-time streaming data
* Improve quantization-aware training

---

## 👨‍💻 Author

**Kinnshuk Bhaduri**

---

## 📜 License

This project is for academic and educational purposes.
