# Quant Model Research

This project is a research module that provides **modular implementations of quantitative models** used in a long-term trading bot.
It supports the bot’s decision-making by simulating market dynamics, forecasting stock performance using deep learning models, and detecting potential bubble risks through a multi-model analytical pipeline.

---

## Repository Structure

| File | Model | Description |
|------|-------|-------------|
| `gbm_simulation_for_stocks.ipynb` | **GBM Simulation** | Generates probabilistic future stock paths and ranks based on expected return |
| `lppl_simulation.ipynb` | **LPPL Model** | Detects bubble behavior and crash timing using log-periodic power law (deterministic + Bayesian) |
| `clam_simulation.ipynb` | **CLAM Model** | Deep learning model (CNN + LSTM + Attention) for multi-ticker price forecasting |

---

## Model Overview

### 1. `Geometric Brownian Motion (GBM)`

Simulates multiple forward-looking price paths for each stock using historical drift ($\mu$) and volatility ($\sigma$):

- **Use Cases**: Stock ranking by average simulated return
- **Modes**: 
  - `quarterly` (65 days)
  - `hourly` (next 7 trading hours)
- **Output**: Ranked list of stocks with visualized simulated paths

---

### 2. `Log-Periodic Power Law (LPPL)`

LPPL captures speculative bubble behavior and predicts **critical crash time ($t_c$)**.

- **Classic LPPL**: Fitted using deterministic methods to detect power-law & log-periodic oscillations  
- **Bayesian MCMC LPPL**: Gives posterior distribution of $t_c$ with HDI (e.g. 94%) + KDE plot  

---

### 3. `CLAM Model (CNN + LSTM + Attention + MLP)`

Trains on 100+ stocks using high-dimensional time-series features (`Open`, `High`, `Low`, `Close`, `Volume`):

- **Features**: Fully custom deep learning model with attention and directional accuracy
- **Modes**: `quarterly` (65 days), `hourly` (7 hours)
- **Outputs**:
  - Predicted future closing prices
  - Expected return per ticker
  - Top 10 stock ranking
  - Accuracy evaluation if actual future prices available

---

## Integration

All models here are **integrated into a larger Long-Term Trading Bot** pipeline which:

- Combines model outputs via weighted scoring
- Uses LLM for sentiment analysis
- Predicts market regime with LPPL
- Makes periodic portfolio selections

#### Refer to the main [Long_Term_Trading](https://github.com/nahniee/Long_Term_Trading) repository for full trading logic & pipeline.

---
