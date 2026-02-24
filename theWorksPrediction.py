import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

def get_live_news_sentiment(ticker):
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock"
    feed = feedparser.parse(rss_url)
    headlines = [entry.title for entry in feed.entries[:10]]
    if not headlines: return 0, []
    results = nlp(headlines)
    sentiment_score = sum([1 if r['label'] == 'Positive' else -1 for r in results if r['label'] != 'Neutral']) / len(headlines)
    return sentiment_score, headlines

def calculate_risk_metrics(data):
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    var_95 = np.percentile(returns, 5)
    return volatility, var_95

def predict_prices(data):
    df = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=0)
    inputs = scaled_data[-60:]
    future_predictions = []
    current_batch = inputs.reshape((1, 60, 1))
    for _ in range(30):
        pred = model.predict(current_batch, verbose=0)
        future_predictions.append(pred[0, 0])
        current_batch = np.append(current_batch[:, 1:, :], [[pred[0]]], axis=1)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

st.set_page_config(layout="wide")
st.title("Pro AI Stock Terminal")
ticker = st.sidebar.text_input("Ticker Symbol", "BTC-USD").upper()

if st.sidebar.button("Generate Full Report"):
    data = yf.download(ticker, period="2y")
    sentiment, news = get_live_news_sentiment(ticker)
    vol, var = calculate_risk_metrics(data)
    predictions = predict_prices(data)
    
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=31)[1:]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment", f"{sentiment:.2f}")
    c2.metric("Annual Volatility", f"{vol:.2%}")
    c3.metric("1-Day VaR (95%)", f"{var:.2%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-100:], y=data['Close'][-100:], name="Actual"))
    fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), name="AI Forecast", line=dict(dash='dot')))
    fig.update_layout(template="plotly_dark", title=f"{ticker} Forecast & Risk Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Latest Market Context"):
        for h in news[:5]: st.write(f"• {h}")