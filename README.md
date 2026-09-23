# Quant Model Research

Research notebooks for the models behind my long-term trading bot. Each one is a self-contained implementation that the bot in [Long_Term_Trading](https://github.com/nahniee/Long_Term_Trading) calls into: a GBM simulation for ranking stocks, an LPPL model for spotting bubbles, and a deep learning forecaster (CLAM).

I later reviewed the GBM and CLAM models in [Signal_Validation](https://github.com/nahniee/Signal_Validation), which led to the weekly CLAM rebuild in `clam_weekly.py`.

## Files

| File | Model | What it does |
|------|-------|--------------|
| `gbm_simulation_for_stocks.ipynb` | GBM | Simulates future price paths and ranks stocks on them |
| `lppl_simulation.ipynb` | LPPL | Looks for bubble behaviour and estimates crash timing, with both a least-squares and a Bayesian fit |
| `clam_simulation.ipynb` | CLAM | CNN + LSTM + attention model that forecasts prices for many tickers |
| `clam_weekly.py`, `clam_weekly.ipynb` | Weekly CLAM | The rebuilt 5-day version, trained one ticker at a time |

## Models

### Geometric Brownian Motion (GBM)

Estimates each stock's drift ($\mu$) and volatility ($\sigma$) from its history and simulates a price path forward. The deployed ranking function draws a single path per stock and scores it by the path's average; the review found that one path adds a lot of noise to the ranking.

- Modes: `quarterly` (63 trading days) and `hourly` (the next 7 trading hours)
- Output: a ranked list of stocks, with plots of the simulated paths

### Log-Periodic Power Law (LPPL)

Fits the log-periodic power law to an index to look for speculative bubbles and estimate the critical crash time $t_c$.

- The classic fit uses least squares to find the power-law trend and the log-periodic oscillation.
- The Bayesian version runs MCMC and gives a posterior for $t_c$, shown as a KDE with a 94% HDI.

### CLAM (CNN + LSTM + attention)

Trained on 94 large-cap stocks using daily Open, High, Low, Close and Volume. The network has a custom attention layer and tracks directional accuracy during training.

- Modes: `quarterly` (65 days ahead) and `hourly` (7 hours ahead)
- Output: forecast prices, expected return per ticker, a top-10 ranking, and an accuracy check when the actual prices are available

## How the bot uses them

The models feed a larger trading pipeline that:

- combines their scores with weights,
- adds news sentiment scored by an LLM,
- uses LPPL to judge the market regime, and
- picks a portfolio on a regular schedule.

The full trading logic is in [Long_Term_Trading](https://github.com/nahniee/Long_Term_Trading).
