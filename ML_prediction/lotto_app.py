"""
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install streamlit pandas numpy plotly statsmodels requests
streamlit run lotto_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dataclasses import dataclass
from statsmodels.tsa.arima.model import ARIMA
import requests
import json

# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class GameConfig:
    name: str
    white_max: int
    white_count: int
    special_max: int
    special_name: str

POWERBALL = GameConfig("Powerball", white_max=69, white_count=5, special_max=26, special_name="Powerball")
MEGAMILLIONS = GameConfig("Mega Millions", white_max=70, white_count=5, special_max=25, special_name="Mega Ball")

REQ_COLS = ["date", "w1", "w2", "w3", "w4", "w5", "special"]
#COLS_FROM_CSV = ["DrawDate","White1","White2","White3","White4","White5","Powerball"]
REQ_COLS = ["date", "w1", "w2", "w3", "w4", "w5", "special"]

COLMAP = {
    "drawdate": "date",
    "white1": "w1",
    "white2": "w2",
    "white3": "w3",
    "white4": "w4",
    "white5": "w5",
    "powerball": "special",   # for Mega Millions you'd map megaball -> special
}

# -----------------------------
# Helpers
# -----------------------------
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file,sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    #df.rename({'DrawDate': 'date', 'White1': 'w1', 'White2': 'w2', 'White3': 'w3', 'White4': 'w4', 'White5': 'w5', 'Powerball': 'special'}, axis=1, inplace=True)
    df = df.rename(columns=COLMAP)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    for c in ["w1", "w2", "w3", "w4", "w5", "special"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Optional columns (if present)
    if "jackpot" in df.columns:
        # remove $ and commas if needed
        df["jackpot"] = (
            df["jackpot"].astype(str)
            .str.replace(r"[$,]", "", regex=True)
        )
        df["jackpot"] = pd.to_numeric(df["jackpot"], errors="coerce")

    df = df.dropna(subset=REQ_COLS).sort_values("date").reset_index(drop=True)
    return df

def validate_ranges(df: pd.DataFrame, game: GameConfig) -> list[str]:
    errs = []
    whites = df[["w1","w2","w3","w4","w5"]].to_numpy()
    special = df["special"].to_numpy()

    if np.any((whites < 1) | (whites > game.white_max)):
        errs.append(f"White balls out of range 1..{game.white_max}")
    if np.any((special < 1) | (special > game.special_max)):
        errs.append(f"{game.special_name} out of range 1..{game.special_max}")

    # Duplicates within a draw (should not happen)
    for i, row in enumerate(whites):
        if len(set(row.tolist())) != game.white_count:
            errs.append(f"Duplicate white balls in row index {i} (date={df.loc[i,'date'].date()})")
            break

    return errs

def dirichlet_smoothed_probs(values: np.ndarray, max_ball: int, alpha: float) -> np.ndarray:
    """
    Returns probabilities for balls 1..max_ball inclusive (index 1..max_ball).
    probs[0] unused.
    """
    counts = np.zeros(max_ball + 1, dtype=float)
    for v in values:
        counts[int(v)] += 1.0
    probs = (counts + alpha) / (counts.sum() + alpha * (max_ball + 1))
    probs[0] = 0.0
    probs = probs / probs.sum()
    return probs

def weighted_sample_without_replacement(probs_1_indexed: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    vals = np.arange(len(probs_1_indexed))
    vals = vals[1:]  # 1..max
    p = probs_1_indexed[1:].astype(float)
    p = p / p.sum()
    pick = rng.choice(vals, size=k, replace=False, p=p)
    pick = sorted(pick.tolist())
    return pick

def fit_frequency_model(df_train: pd.DataFrame, game: GameConfig, alpha: float):
    whites = df_train[["w1","w2","w3","w4","w5"]].to_numpy().reshape(-1)
    special = df_train["special"].to_numpy()

    p_white = dirichlet_smoothed_probs(whites, game.white_max, alpha=alpha)
    p_special = dirichlet_smoothed_probs(special, game.special_max, alpha=alpha)
    return p_white, p_special

def generate_picks(p_white, p_special, game: GameConfig, n: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        whites = weighted_sample_without_replacement(p_white, game.white_count, rng)
        special = int(rng.choice(np.arange(1, game.special_max + 1), p=p_special[1:]))
        rows.append(whites + [special])
    cols = [f"w{i}" for i in range(1, game.white_count + 1)] + [game.special_name]
    return pd.DataFrame(rows, columns=cols)

def backtest_logloss(df: pd.DataFrame, game: GameConfig, alpha: float, lookback: int):
    """
    Backtest a simple frequency model:
    train on previous draws, score next draw by product of per-number probabilities (approx).
    Compare to uniform baseline.
    """
    if len(df) <= lookback + 5:
        return None

    ll_model = []
    ll_uniform = []
    white_uniform = 1.0 / comb(game.white_max, game.white_count)
    special_uniform = 1.0 / game.special_max
    p_uniform_draw = white_uniform * special_uniform
    log_uniform = np.log(p_uniform_draw)

    for i in range(lookback, len(df)):
        train = df.iloc[i - lookback:i]
        test = df.iloc[i]
        p_white, p_special = fit_frequency_model(train, game, alpha)

        whites = test[["w1","w2","w3","w4","w5"]].to_numpy(dtype=int)
        special = int(test["special"])

        # Approx likelihood (not exact combinatorial model, but consistent for comparison)
        p = np.prod([p_white[w] for w in whites]) * p_special[special]
        p = max(p, 1e-300)
        ll_model.append(np.log(p))
        ll_uniform.append(log_uniform)

    return {
        "avg_loglik_model": float(np.mean(ll_model)),
        "avg_loglik_uniform": float(np.mean(ll_uniform)),
        "diff_model_minus_uniform": float(np.mean(ll_model) - np.mean(ll_uniform)),
        "n_scored": len(ll_model),
    }

def comb(n, k):
    # Simple nCk without importing scipy
    from math import prod
    k = min(k, n - k)
    if k < 0:
        return 0
    if k == 0:
        return 1
    num = prod(range(n, n - k, -1))
    den = prod(range(1, k + 1))
    return num // den

def try_jackpot_forecast(df: pd.DataFrame, steps: int = 5):
    if "jackpot" not in df.columns:
        return None
    s = pd.to_numeric(df["jackpot"], errors="coerce").dropna()
    if len(s) < 20:
        return None
    # ARIMA on log(jackpot) for stability
    y = np.log(s.astype(float).values)
    model = ARIMA(y, order=(1, 1, 1)).fit()
    fc = model.forecast(steps=steps)
    return np.exp(fc)

# -----------------------------
# Optional LLM report
# -----------------------------
def llm_report_openai(api_key: str, summary: dict) -> str:
    # Uses OpenAI Responses API style via HTTPS (no openai package required)
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"""
You are a data analyst. Summarize the following lottery draw analysis.
Be clear that lottery draws are random and this does not predict winners.
Give short bullet points: trends, hot numbers, caveats, and how to use the app responsibly.

ANALYSIS_JSON:
{json.dumps(summary, indent=2)}
""".strip()

    payload = {
        "model": "gpt-4.1-mini",
        "input": prompt
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Extract text
    out = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out.append(c.get("text", ""))
    return "\n".join(out).strip()

def llm_report_ollama(model: str, summary: dict) -> str:
    prompt = f"""
Summarize this lottery draw analysis. State clearly it cannot predict winning numbers.
Return concise bullets.

{json.dumps(summary, indent=2)}
""".strip()
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Powerball / Mega Millions Data App", layout="wide")
st.title("Powerball / Mega Millions — Data-Driven Analyzer (ML + optional LLM report)")

st.caption(
    "Educational analytics only. Lottery outcomes are designed to be random; this app does not predict winning numbers."
)

with st.sidebar:
    game_name = st.selectbox("Game", [POWERBALL.name, MEGAMILLIONS.name])
    game = POWERBALL if game_name == POWERBALL.name else MEGAMILLIONS

    alpha = st.slider("Smoothing (alpha)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    lookback = st.slider("Backtest lookback draws", min_value=20, max_value=500, value=150, step=10)
    n_picks = st.number_input("Generate picks", min_value=1, max_value=50, value=10, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

    st.markdown("---")
    st.subheader("Optional LLM report")
    llm_mode = st.selectbox("LLM mode", ["Off", "OpenAI API Key", "Ollama (local)"])
    openai_key = st.text_input("OpenAI API key", type="password") if llm_mode == "OpenAI API Key" else ""
    ollama_model = st.text_input("Ollama model name", value="llama3.1") if llm_mode == "Ollama (local)" else ""

uploaded = st.file_uploader("Upload historical draws CSV", type=["csv"])
uploaded_jackpot = st.file_uploader("Optional: Upload jackpot CSV (DrawDate,Jackpot)", type=["csv"])

if not uploaded:
    st.info("Upload a CSV with columns: date,w1,w2,w3,w4,w5,special (optional: multiplier,jackpot)")
    st.stop()

try:
    df = load_data(uploaded)
    if uploaded_jackpot is not None:
        jp = pd.read_csv(uploaded_jackpot, sep=None, engine="python")
        jp.columns = [c.strip().lower() for c in jp.columns]
        jp = jp.rename(columns={"drawdate": "date", "jackpot": "jackpot"})

        jp["date"] = pd.to_datetime(jp["date"])
        jp["jackpot"] = (
            jp["jackpot"].astype(str).str.replace(r"[$,]", "", regex=True)
        )
        jp["jackpot"] = pd.to_numeric(jp["jackpot"], errors="coerce")

        df = df.merge(jp[["date", "jackpot"]], on="date", how="left")
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

errs = validate_ranges(df, game)
if errs:
    st.error("Data validation issues:\n- " + "\n- ".join(errs))
    st.stop()

st.subheader("Data preview")
st.dataframe(df.tail(20), use_container_width=True)

# -----------------------------
# EDA: frequencies
# -----------------------------
st.subheader("Frequency analysis")

whites = df[["w1","w2","w3","w4","w5"]].to_numpy().reshape(-1)
special = df["special"].to_numpy()

white_counts = pd.Series(whites).value_counts().sort_index()
special_counts = pd.Series(special).value_counts().sort_index()

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(x=white_counts.index, y=white_counts.values, labels={"x":"White ball", "y":"Count"},
                 title=f"{game.name}: White ball frequencies")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.bar(x=special_counts.index, y=special_counts.values, labels={"x":game.special_name, "y":"Count"},
                 title=f"{game.name}: {game.special_name} frequencies")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Fit model + generate picks
# -----------------------------
p_white, p_special = fit_frequency_model(df, game, alpha=alpha)
picks = generate_picks(p_white, p_special, game, n=int(n_picks), seed=int(seed))

st.subheader("Generated picks (weighted random; not predictions)")
st.dataframe(picks, use_container_width=True)

# -----------------------------
# Backtest
# -----------------------------
st.subheader("Backtest (model vs uniform baseline)")
bt = backtest_logloss(df, game, alpha=alpha, lookback=int(lookback))
if bt is None:
    st.warning("Not enough data to backtest with the selected lookback.")
else:
    st.json(bt)
    st.caption("If 'diff_model_minus_uniform' is near 0, the model is not better than uniform randomness (expected for lottery data).")

# -----------------------------
# Jackpot forecast (optional)
# -----------------------------
st.subheader("Jackpot forecast (optional, if jackpot column exists)")
fc = try_jackpot_forecast(df, steps=5)
if fc is None:
    st.info("No usable 'jackpot' column found (or insufficient data). Add a numeric jackpot column to enable forecasting.")
else:
    st.write("Next 5-step forecast (same units as your jackpot column):")
    st.dataframe(pd.DataFrame({"forecast": fc}), use_container_width=True)

# -----------------------------
# LLM summary (optional)
# -----------------------------
summary = {
    "game": game.name,
    "n_draws": int(len(df)),
    "date_min": str(df["date"].min().date()),
    "date_max": str(df["date"].max().date()),
    "top_white": white_counts.sort_values(ascending=False).head(10).to_dict(),
    "top_special": special_counts.sort_values(ascending=False).head(10).to_dict(),
    "backtest": bt,
}

if llm_mode != "Off":
    st.subheader("LLM-generated report")
    try:
        if llm_mode == "OpenAI API Key":
            if not openai_key:
                st.warning("Enter an OpenAI API key in the sidebar.")
            else:
                st.write(llm_report_openai(openai_key, summary))
        elif llm_mode == "Ollama (local)":
            st.write(llm_report_ollama(ollama_model, summary))
    except Exception as e:
        st.error(f"LLM report failed: {e}")
