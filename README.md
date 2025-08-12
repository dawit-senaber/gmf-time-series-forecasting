# 📈 Portfolio Optimization with Time Series Forecasting

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)]()

---

## 📌 Overview
This project delivers an **end-to-end financial portfolio optimization pipeline** using **time series forecasting** and **Modern Portfolio Theory**.  
Developed for **GMF Investments**, it integrates **LSTM deep learning models** with advanced risk metrics to optimize allocation between:
- Tesla (**TSLA**)
- Vanguard Total Bond Market ETF (**BND**)
- S&P 500 ETF (**SPY**)

---

## 🎯 Business Objective
> *"Help clients achieve financial objectives by minimizing risks and capitalizing on market opportunities through data-driven portfolio optimization."*

---

## 🚀 Key Features
- **Advanced Forecasting** – LSTM models with **MAE: 63.27**
- **Portfolio Optimization** – Efficient Frontier with allocation constraints
- **Risk Management** – VaR, volatility, and drawdown analysis
- **Backtesting** – Strategy validation with transaction costs
- **Automated Reporting** – PDF investment memos with charts & insights

---

## 🏗 Solution Architecture

[ Data Acquisition ] → [ Preprocessing ] → [ Forecasting Models ]
↓ ↓ ↓
[ Optimization Engine ] → [ Backtesting ] → [ Reporting ]


---

## 📥 Installation

# Clone repository
- git clone https://github.com/dawit-senaber/gmf-time-series-forecasting.git
- cd gmf-time-series-forecasting

# Create virtual environment
python -m venv gmf-env

# Activate environment
# Linux/Mac:
source gmf-env/bin/activate
# Windows:
.\gmf-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

▶ Usage

# Run full pipeline
python main.py


📊 Key Results

| Metric           | Optimized Portfolio | 60/40 Benchmark |
| ---------------- | ------------------- | --------------- |
| **Total Return** | 44.11%              | 46.20%          |
| **Sharpe Ratio** | 1.01                | 1.04            |
| **Max Drawdown** | -22.4%              | -15.2%          |


Efficient Frontier:

https://github.com/dawit-senaber/gmf-time-series-forecasting/blob/main/reports/figures/efficient_frontier.png
Optimal portfolios on the Efficient Frontier

📂 Project Structure

gmf-time-series-forecasting/
├── data/
│   ├── raw/               # Raw financial data
│   └── processed/         # Cleaned datasets
├── models/                # Trained model artifacts
├── reports/
│   ├── figures/           # Visualizations
├── src/
│   ├── data_loader.py     # Data acquisition
│   ├── preprocessing.py   # Data cleaning
│   ├── forecasting.py     # ARIMA/LSTM models
│   ├── optimization.py    # Portfolio optimization
│   ├── backtesting.py     # Strategy validation
│   └── reporting.py       # PDF generation
├── config.py              # Global parameters
├── main.py                # Execution pipeline
├── requirements.txt       # Dependencies
└── README.md              # This document


⚙ Technical Specifications

LSTM Forecasting Model

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])

Optimization Constraints

ef.add_constraint(lambda w: w[0] <= 0.3)  # Max 30% TSLA
ef.add_constraint(lambda w: w[1] >= 0.05) # Min 5% BND


📜 License
This project is licensed under the MIT License – see the LICENSE file for details.

Dawit Senaber | August 2025
Data-Driven Investment Strategies