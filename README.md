# 📈 theWorksPrediction 2026
### *Hybrid Intelligence: Sentiment Analysis + LSTM Time-Series Forecasting*

A high-performance financial analysis tool built in Python. This terminal bridges the gap between **quantitative data** (price action) and **qualitative data** (market sentiment) to provide a holistic 30-day stock price forecast.

---

## 🏗️ System Architecture
Below is the data flow and model hierarchy used to generate predictions.

<img width="1024" height="1024" alt="Gemini_Generated_Image_v8j4z2v8j4z2v8j4" src="https://github.com/user-attachments/assets/ce749043-ac4b-4cf6-adde-59f675a6a760" />



---

## 🚀 Key Features
* **Real-Time Sentiment:** Scrapes the latest 10 global headlines and processes them through **FinBERT**.
* **Deep Learning Forecast:** Uses a stacked **LSTM (Long Short-Term Memory)** network.
* **Risk Engine:** Calculates **Annualized Volatility** and **Value at Risk (VaR)**.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Frontend** | Streamlit |
| **Data Source** | yfinance, Feedparser |
| **AI Models** | TensorFlow, Hugging Face (FinBERT) |

---

## 📥 Quick Start
1. **Install Dependencies:**
   `pip install -r requirements.txt`
2. **Run App:**
   `streamlit run app.py`

---

## 📊 Methodology


### 1. The Sentiment Engine
Uses **FinBERT** to identify "Bullish," "Bearish," or "Neutral" sentiment from live RSS feeds.

### 2. The LSTM Predictor
Analyzes a **60-day sliding window** of historical closing prices to project the next 30 days of movement.

### 3. Risk Metrics
Includes **Volatility** and **VaR (95%)** to provide a statistically likely "worst-case scenario" for risk management.

---

## 📜 License & Usage Terms
**Copyright (c) 2026 Soham Pattanayek. All Rights Reserved.**

This code is provided for **educational and analytical purposes only**.
1. **Attribution:** Any display or demo must credit **Soham Pattanayek**.
2. **Non-Commercial Use:** You may not use this code for financial gain.
3. **No Redistribution:** You are **strictly prohibited** from copying, modifying, or redistributing the source code without prior written consent.

---

## ⚠️ Disclaimer
**Not Financial Advice.** AI models can hallucinate trends. Always perform your own due diligence.
