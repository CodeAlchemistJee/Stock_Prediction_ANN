# 📈 Deep Learning for Stock Market Prediction (Feedforward ANN)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg?logo=scikit-learn)
![Finance](https://img.shields.io/badge/yfinance-Live_Data-success.svg)

A complete end-to-end Machine Learning pipeline that dynamically ingests live financial data via the `yfinance` API, preprocesses time-series data using Sliding Windows, and forecasts future stock prices using a multi-layer Feedforward Artificial Neural Network (ANN) built with TensorFlow/Keras.

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Deep-Dive Analysis & Results](#deep-dive-analysis--results)
4. [Installation & Usage](#installation--usage)

## 🎯 Project Overview
Traditional technical analysis struggles to capture the highly volatile, non-linear patterns of modern financial markets. The objective of this project is to leverage Deep Learning to map these complex relationships. The system fetches chronological market data, scales it to ensure gradient stability, and trains a customized neural network to recognize historical momentum and project future asset valuations.

## 🧠 System Architecture

### 1. Data Pipeline (Sliding Window)
To give a standard dense network the concept of "time", the historical data is restructured using a **60-Day Sliding Window** technique. The model takes the previous 60 trading days as features (Input $X$) to predict the 61st day's closing price (Target $y$).

### 2. Neural Network Topology
* **Input Layer:** 60 nodes (representing the lookback window).
* **Hidden Layer 1:** 64 nodes utilizing Rectified Linear Unit (ReLU) activation.
* **Dropout Layer:** A 20% dropout rate (`0.2`) is applied to randomly deactivate neurons during training, preventing the model from mathematically overfitting to training noise.
* **Hidden Layer 2:** 32 nodes (ReLU).
* **Output Layer:** 1 node outputting a continuous float (the predicted USD price).
* **Optimization:** `Adam` optimizer minimizing Mean Squared Error (MSE) via backpropagation.

## 📊 Deep-Dive Analysis & Results

The model's performance was evaluated chronologically on unseen testing data (an 80/20 train-test split) yielding the following metrics:
* **Root Mean Squared Error (RMSE):** ~$21.20
* **Mean Absolute Percentage Error (MAPE):** ~11.61%

### The Extrapolation Bias in Dense Networks
*(Include your Matplotlib chart here in the actual repo by dropping the image into the README!)*

Upon visual inspection of the actual vs. predicted curves, the model demonstrated **exceptional pattern recognition**—perfectly mirroring rapid market spikes and dips based on the 60-day momentum. 

However, the model exhibits a consistent downward vertical shift. Because standard Feedforward networks lack long-term sequential memory states (unlike LSTMs), they suffer from **Extrapolation Bias**. When evaluated on recent testing data where the baseline price is historically high, the model correctly predicts the directional movement but anchors its baseline predictions to the lower absolute prices it memorized during the 2020-2022 training period.

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Stock-Prediction-ANN.git](https://github.com/CodeAlchemistJee/Stock-Prediction-ANN.git)
   cd Stock-Prediction-ANN
