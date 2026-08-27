# Video Explanation Script — Crime Incident Prediction Project

**Target length: 16–18 minutes** (the brief requires ≥15 min and ≤25 min). Timings are guides; speak naturally and don't rush. Have the notebook open and scroll to the relevant cell as you narrate each section. Lines in *italics* are stage directions, not things to read aloud.

---

## 1 · Introduction — 1.5 min

*Screen: title / project summary cell.*

"Hi, I'm [your name], and this is my walkthrough of the Crime Incident Prediction project.

This is a **supervised machine-learning regression project**. The business problem is simple to state: a police force has limited patrols, and it needs to know *where* and *when* crime will happen so it can put those patrols in the right place ahead of time. My model predicts the **number of crime incidents for each neighbourhood, each crime type, each month**, using thirteen years of incident records.

Who benefits? Police operations get data-driven patrol scheduling; city planners get guidance on where to invest in lighting, cameras and prevention; and the wider community gets safer streets.

My tech stack is Python, with pandas and NumPy for data work, scikit-learn and XGBoost for the machine-learning models, statsmodels for the time-series model, and Matplotlib and Seaborn for the visuals. I'll also describe how this deploys with Streamlit and a GenAI layer at the end."

---

## 2 · Problem understanding — 2 min

*Screen: Section 1 markdown.*

"Let me be precise about what the model does. It's a **forecasting / regression** task — the output is a *count*, how many incidents, not a category.

Why does this problem matter? Because crime is not spread evenly. Some neighbourhoods carry far more risk than others, and crime rises and falls with the seasons. A raw log of past incidents doesn't tell a commander how much to expect next month, so without a forecast, policing is reactive — it chases last week's news instead of anticipating next month's load.

I deliberately predict at the **neighbourhood-and-crime-type level** rather than one city-wide number, because that's what makes the output *actionable*. A single total can't tell you to move a car from one district to another, or that vehicle theft spikes in summer while residential break-ins stay flat. Predicting per group keeps exactly the spatial and seasonal detail a deployment decision needs.

And why machine learning? Because the link between time, place and crime volume is non-linear, seasonal, and full of interactions — patterns a hand-written rule can't capture, but a model can learn straight from the history."

---

## 3 · Data understanding & EDA — 2.5 min

*Screen: Section 3 and 4 — scroll through the EDA charts as you talk.*

"The dataset is **474,565 individual incidents from 1999 to 2011** — nine crime types across twenty-four neighbourhoods. Each row has the what, the where — including two coordinate systems — and the when, down to the hour and minute.

*Point at the 'by type' chart.* First, the crime mix is very concentrated. Theft from Vehicle alone is about a third of everything, and the top three property crimes dominate. That concentration is the regression version of class imbalance — I'll come back to it.

*Point at the neighbourhood chart.* Crime is also concentrated in *space* — the busiest neighbourhoods record many times the volume of the quietest. That's the 'where' signal, and the reason a per-neighbourhood model is worth building.

*Point at the trend chart.* Over the thirteen years there's a clear **downward trend** with regular yearly waves. The trend means I can't assume the current level just continues; the waves are seasonality I can exploit.

*Point at the seasonality bars.* Confirmed here — incidents climb through summer and dip in winter, every year.

Now the most important data-quality finding. About 49,000 rows are **missing the hour and minute**. When I checked *which* rows, they were **every single one 'Offence Against a Person'** — the timestamp is deliberately censored for those sensitive cases. That matters because the naïve move would be to fill the missing hour with an average — which would fabricate a time pattern for exactly the crime where the real time was intentionally hidden. So I excluded them from the hour analysis instead. Spotting that prevented a real mistake."

---

## 4 · Data preprocessing & feature preparation — 3 min

*Screen: Section 5 — the panel-building and feature cells.*

"Here's the key transformation. The raw file is a log of *events*, but my target is a monthly *count per group*. So I reshaped it into a **spatio-temporal panel**: one row for every neighbourhood–type–month combination across the full span.

A subtle but critical step: **zero-filling**. If a neighbourhood-type had no incidents in a month, that month is simply *absent* from the raw log. If I left it out, the model would never learn that quiet combinations exist. So I built the complete grid and filled genuine absences with zero — a real, informative value.

Then the features, and every one earns its place:

- **Cyclical month encoding** — sine and cosine — because December and January are next to each other on the calendar but far apart as the numbers 12 and 1. This is how the seasonal wave becomes learnable.
- **Lag features** — last month, and the same month a year ago — because crime has momentum.
- **Rolling means and standard deviation** — a smoothed recent level and its volatility.

Crucially, every lag and rolling feature uses only *past* values — I shift by one month before rolling — so **no future information leaks** into any row. For a forecasting problem that discipline is everything.

I also encode neighbourhood and type as integer codes, which tree models handle natively.

*Screen: the correlation heatmap.* I checked multicollinearity too. The recent-history features are correlated, as expected — but that's harmless for tree models, which are invariant to it. I flag it only because it would matter for interpreting a linear model's coefficients.

Finally, the target. *Screen: Section 6.* The brief asks about class imbalance. My target is a count, so the equivalent is **skew and zero-inflation** — and it's strongly right-skewed, skew around 7.6, with about a quarter of group-months at zero. So I **train on log-of-count**, which makes the distribution near-symmetric, and I invert it at prediction time. I also lead with **MAE** as my metric, because unlike RMSE it isn't dominated by the few extreme groups — it's the honest 'typical error'."

---

## 5 · Model building & evaluation — 3.5 min

*Screen: Section 7 and 8 — the results table and MAE bar chart.*

"I compared two complementary families, because the stakeholders work at two granularities.

First, **supervised regression** on the panel: Linear Regression as an interpretable baseline, then Random Forest and XGBoost — non-linear ensembles that capture interactions and seasonality. Second, a **SARIMA** time-series model on the city-wide total for the aggregate planning view. And I included a **naïve 'same as last month'** baseline, because beating a naïve baseline is the real test of value.

On validation — this is important — I did **not** use a random split, because that would leak the future into the past. I trained on 1999 through 2010 and tested on the entire unseen 2011, exactly the situation the model faces in production.

*Point at the results table.* Here are the numbers. **Random Forest is the best operational model** — mean absolute error about **3.2 incidents**, R-squared **0.94** on a full unseen year. XGBoost is right behind.

Now, I want to be honest about the naïve baseline — it's genuinely strong, also around 0.94, because monthly counts are highly autocorrelated. That's not a disappointment; it's honest evaluation. The machine-learning models still beat it on MAE, and they add something the baseline can't: one model that generalises across *every* neighbourhood-type and encodes seasonality, so it stays reliable when a group's recent history is noisy or trending. Linear Regression, by contrast, lags badly at R-squared 0.25 — proof the relationship is non-linear, which is exactly why the tree ensembles win.

*Screen: SARIMA forecast chart.* For the **city-wide total**, SARIMA forecasts the level much better than the granular model summed up, because small per-group errors accumulate when you add them together. So the lesson is *right tool for the right granularity*: SARIMA for the planner's total, Random Forest for the commander's neighbourhood map. They're complementary, not competing."

---

## 6 · Explainability & deployment — 2 min

*Screen: Section 9 — feature importance chart.*

"A model police will act on has to be interpretable, so I read feature importance from both tree models — and they agree.

The top signals are **recent history** — the rolling means and last month's count. A group's own recent level is the best guide to next month, which matches police intuition. **Seasonality** shows up next through the cyclical month and the twelve-month lag, letting the model raise summer estimates *before* the rise. Neighbourhood and type rank lower — not because location doesn't matter, but because the lag features already carry each group's characteristic level. So it's a clean, defensible story, not a black box.

*Screen: Section 10.* On deployment: the model is small and fast. I'd wrap it in a **Streamlit app** where a commander picks a neighbourhood, type and month and gets a predicted count with a history chart and a hotspot map. A **GenAI layer**, like Gemini, turns that number into a plain-English shift briefing an officer will actually read. And **Azure ML** could host it with scheduled monthly retraining. The training code saves the model to a file, so it drops straight into any of those."

---

## 7 · Challenges, optimisation & improvements — 1.5 min

"A few honest challenges. The biggest was **framing** — getting from an event log to a zero-filled panel without leaking the future took real care, and it mattered more than any algorithm choice. Preventing leakage in the rolling features and using a future hold-out were the key optimisations. The **strong naïve baseline** was a challenge of a different kind — it forced me to be honest about how much the model really adds.

For future improvements: the data ends in 2011, so I'd **retrain on current records** before any deployment. I'd add **external drivers** — weather, holidays, events, socioeconomic data — which probably explain the variance the lags miss. I'd try **hurdle models** for the zero-inflated quiet groups, run **hyperparameter search**, and add **quantile predictions** so commanders get a range, not just a point. And I'd bring the coordinates back with a proper **spatial model** for block-level resolution."

---

## 8 · Learnings & business impact — 1.5 min

*Screen: Section 11 conclusion.*

"To wrap up — the business impact. Operations can shift patrols toward the neighbourhood-types and months the model flags as rising, *before* they rise: proactive instead of reactive. Planning can direct lighting, cameras and prevention spend to the areas the data names as persistently high-risk. And because the model is explainable, those decisions can be justified to the public.

What I learned: framing the problem well mattered more than the algorithm; a naïve baseline is essential to keep yourself honest — without it, a 0.94 R-squared would have *looked* like a triumph instead of a modest, real gain; and reading the missingness in the data prevented a genuine modelling error.

The current limitations are the 2011 cutoff and the lack of external features — both clear next steps. Thanks for watching; the full reproducible notebook and the GitHub repo are linked in the summary."

---

### Quick prep checklist before recording
- Notebook fully run top-to-bottom, all charts visible.
- GitHub placeholder link replaced with your real repo.
- Screen readable at recording resolution; zoom the browser to ~110%.
- Do a timing pass — aim to land between 16 and 18 minutes.
