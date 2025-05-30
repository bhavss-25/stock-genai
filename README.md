# 🤖 AI Stock Price Predictor

A powerful, interactive Streamlit web application that leverages machine learning models (Random Forest and Gradient Boosting) to predict the **next day's closing stock price** based on user-provided OHLC (Open, High, Low, Close) values and technical indicators.

![App Screenshot](https://img.shields.io/badge/Built%20with-Streamlit-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/github/license/yourusername/stock-price-predictor?style=for-the-badge)

---

## 📌 Features

- 📊 **User-Friendly Web Interface** built with Streamlit
- ⚙️ Accepts real-time **OHLC input**
- 🤖 Predicts **next day's closing price** using:
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
- 📉 Displays:
  - Predicted Price
  - Confidence Interval
  - Price Direction
  - Volatility-based Risk Level
- 🧠 Performs **Technical Analysis**:
  - Candlestick interpretation
  - Shadow and body insights
  - Volatility metrics

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor
