# 🚔 Crime Incident Prediction — FBI Crime Investigation Project

Predicting the **number of crime incidents per neighbourhood, per crime type, per month** from 13 years of granular incident records, to help police forces and city planners allocate resources *before* crime happens rather than after.

> **Best model:** Random Forest — **MAE ≈ 3.25 incidents**, **R² ≈ 0.94** on a fully unseen hold-out year (2011).

---

## 📌 Problem

Police have finite patrols and must decide where and when to deploy them. A raw incident log doesn't say how much crime to expect next month or where. This project turns **474,565 incident records (1999–2011)** into a monthly, per-neighbourhood, per-type forecast — a **supervised regression** task that preserves exactly the spatial and seasonal detail a deployment decision needs.

**Who benefits:** police operations (patrol scheduling), city planners (lighting/cameras/prevention), and communities (safer streets).

> **Data note:** the records are the open **Vancouver Police Department** crime dataset (the brief calls it "FBI"). The methodology is city-agnostic.

---

## 🧰 Tech stack

| Area | Tools |
|---|---|
| Data & compute | Python 3, pandas, NumPy |
| ML | scikit-learn (LinearRegression, RandomForest), XGBoost |
| Time series | statsmodels (SARIMA) |
| Visualisation | Matplotlib, Seaborn |
| Deployment (design) | Streamlit + GenAI (Gemini), optional Azure ML |

---

## 🔬 Approach

1. **Reshape** the event log into a **spatio-temporal panel** — one row per `(neighbourhood, type, month)`, with genuine absences **zero-filled**.
2. **Engineer temporal-memory features** — lags (1, 2, 3, 12), rolling mean/std, cyclical month (sin/cos), trend index — all leak-free (past values only).
3. **Condition the target** — the count is right-skewed (skew ≈ 7.6, ~24% zeros), so train on `log1p(count)`; report **MAE** as the headline metric.
4. **Validate on the future** — train 1999–2010, test on the entire unseen **2011** (no random split → no leakage).
5. **Compare models** — Linear / Random Forest / XGBoost + a **naïve last-month baseline**, plus **SARIMA** on the city-wide series.

---

## 📊 Results (2011 hold-out)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naïve (last month) | 3.71 | 7.09 | 0.942 |
| Linear Regression | 9.21 | 25.59 | 0.245 |
| **Random Forest** | **3.25** | **7.21** | **0.940** |
| XGBoost | 3.41 | 8.57 | 0.915 |
| SARIMA *(city-wide)* | 216.9 | 294.4 | 0.079 |

**Takeaways**
- **Random Forest wins** on the operational per-group task (lowest MAE).
- The **naïve baseline is strong** (counts are autocorrelated) — an honest evaluation. The ML models beat it on MAE *and* generalise across every group with one seasonal model.
- **Right tool per granularity:** SARIMA is better for the *city-wide aggregate*; tree models for *per-neighbourhood-per-type* detail.
- **Feature importance:** recent history (rolling means, `lag_1`) dominates, then seasonality (`month_cos`, `lag_12`) — interpretable and defensible.

---

## 🗂️ Repository structure

```
.
├── Crime_Prediction_Project.ipynb   # main notebook (EDA → features → models → evaluation → conclusion)
├── Video_Explanation_Script.md      # 16–18 min narration script for the submission video
├── requirements.txt                 # pinned dependencies
├── LICENSE                          # MIT
├── README.md                        # this file
└── data/
    └── Train.xlsx                    # place the dataset here (see data/README.md)
```

---

## ▶️ How to run

**Google Colab (recommended)**
1. Open `Crime_Prediction_Project.ipynb` in Colab.
2. Upload `Train.xlsx` (or `Train.xlsb`) via the file browser.
3. Uncomment the `pip install` line in Section 0, then **Runtime → Run all**.

**Locally**
```bash
git clone https://github.com/harshr002/crime-incident-prediction.git
cd crime-incident-prediction
pip install -r requirements.txt
# put Train.xlsx (or Train.xlsb) in the data/ folder, then:
jupyter notebook Crime_Prediction_Project.ipynb
```

---

## 🚀 Deployment path (design)

- **Streamlit** app: pick neighbourhood + type + month → predicted count, history chart, hotspot map.
- **GenAI (Gemini)** layer: turn the number into a plain-English shift briefing.
- **Azure ML**: host the model, schedule monthly retraining, expose a REST endpoint. Model serialises with `joblib.dump(rf, "model.pkl")`.

---

## 🔭 Future work

Retrain on current data (this set ends 2011) · add exogenous drivers (weather, holidays, events, socioeconomic) · hurdle/zero-inflated models for quiet groups · hyperparameter search + quantile (range) predictions · spatial modelling with GeoPandas for block-level resolution.

---

## 📄 License

Educational project. Dataset © City of Vancouver / VPD open data.
