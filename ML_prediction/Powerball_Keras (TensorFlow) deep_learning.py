# Nice — below is a complete, well-documented Keras (TensorFlow) deep-learning pipeline that:
# 
# Loads historical Powerball CSV data (expects columns DrawDate,White1,White2,White3,White4,White5,Powerball).
# 
# Encodes draws as sequences of multi-hot vectors (whites) + one-hot (red).
# 
# Trains an LSTM sequence model to predict the next draw: a multi-label output for the 5 white balls (sigmoid over 69 outputs) and a softmax for the red ball (26 classes).
# 
# Shows how to sample candidate ticket combinations from model output (top-k + temperature sampling) while enforcing valid constraints (5 unique white balls, range limits).
# 
# Includes helper functions to verify a combination against historical data.
# 
# Important reminder: This is an educational/data-science project. Powerball draws are random. This model cannot actually predict a jackpot — it only learns statistical patterns in historical data. Use responsibly.
# 
# """
# powerball_lstm_predictor.py
# 
# Requirements:
# - Python 3.8+
# - pandas, numpy
# - tensorflow (2.x)
# - scikit-learn
# 
# Install example:
# pip install pandas numpy scikit-learn tensorflow
# """

import os
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ---------------------------
# Config
# ---------------------------
CSV_PATH = r'C:\Users\ankit\Desktop\JupyterProjects\Python\ML_prediction\powerball_history.csv'
# CSV_PATH = "powerball_history.csv"   # expected columns: DrawDate, White1..White5, Powerball
SEQ_LEN = 20                         # number of past draws to use as input
BATCH_SIZE = 64
EPOCHS = 80
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MAX_WHITE = 69
MAX_RED = 26

# ---------------------------
# Utilities: encoding/decode
# ---------------------------

def encode_draw_as_vector(white_list, red_value):
    """
    Encode a single draw as concatenated vector:
      - whites: 69-d binary (index 0 represents ball 1)
      - red: 26-d one-hot (index 0 = red ball 1)
    Returns vector length 69+26 = 95
    """
    whites_vec = np.zeros(MAX_WHITE, dtype=np.float32)
    for w in white_list:
        if 1 <= w <= MAX_WHITE:
            whites_vec[w - 1] = 1.0
        else:
            raise ValueError(f"White ball {w} out of range 1-{MAX_WHITE}")
    red_vec = np.zeros(MAX_RED, dtype=np.float32)
    if 1 <= red_value <= MAX_RED:
        red_vec[red_value - 1] = 1.0
    else:
        raise ValueError(f"Red ball {red_value} out of range 1-{MAX_RED}")
    return np.concatenate([whites_vec, red_vec])

def decode_whites_from_probs(probs, top_k=5):
    """
    probs: array length 69, values in [0,1]
    returns top_k white numbers (1-indexed)
    """
    idx = np.argsort(probs)[-top_k:][::-1]
    return [int(i + 1) for i in idx]

def decode_red_from_probs(probs, temperature=1.0):
    """
    probs: raw logits or probabilities length 26.
    We apply softmax (with temperature) and sample or choose argmax.
    """
    # Turn into probabilities if they are logits or not summing to 1
    p = np.array(probs, dtype=np.float64)
    # temperature sampling
    p = np.log(np.clip(p, 1e-9, 1-1e-9)) / (temperature + 1e-12)
    exp_p = np.exp(p - np.max(p))
    p = exp_p / np.sum(exp_p)
    return int(np.argmax(p) + 1)  # return 1-indexed

# ---------------------------
# Load & preprocess data
# ---------------------------
def load_and_prepare(csv_path=CSV_PATH, seq_len=SEQ_LEN):
    df = pd.read_csv(csv_path)
    # Basic sanitize and sorting
    if 'DrawDate' in df.columns:
        df['DrawDate'] = pd.to_datetime(df['DrawDate'], errors='coerce')
        df = df.sort_values('DrawDate').reset_index(drop=True)
    else:
        # If no date column, assume existing order is chronological
        df = df.reset_index(drop=True)
    # Ensure integer types
    for c in ['White1','White2','White3','White4','White5','Powerball']:
        df[c] = df[c].astype(int)

    # Build encoded vectors
    encoded = []
    draws = []
    for _, row in df.iterrows():
        whites = [row['White1'], row['White2'], row['White3'], row['White4'], row['White5']]
        red = int(row['Powerball'])
        encoded.append(encode_draw_as_vector(whites, red))
        draws.append({
            'date': row['DrawDate'] if 'DrawDate' in df.columns else None,
            'whites': whites,
            'red': red
        })
    encoded = np.stack(encoded)  # shape (n_draws, 95)

    # Build sequences
    X, y_whites, y_red = [], [], []
    for i in range(seq_len, len(encoded)):
        seq = encoded[i - seq_len:i]   # shape (seq_len, 95)
        X.append(seq)
        # target = next draw (i)
        target_vec = encoded[i]
        y_whites.append(target_vec[:MAX_WHITE])   # binary vector
        y_red.append(target_vec[MAX_WHITE:])      # one-hot vector
    X = np.array(X, dtype=np.float32)
    y_whites = np.array(y_whites, dtype=np.float32)
    y_red = np.array(y_red, dtype=np.float32)
    return X, y_whites, y_red, draws, df

# ---------------------------
# Model architecture
# ---------------------------
def build_model(seq_len=SEQ_LEN, input_dim=MAX_WHITE + MAX_RED, lstm_units=256, dense_units=128, dropout_rate=0.2):
    inp = Input(shape=(seq_len, input_dim), name='input_sequence')
    x = LSTM(lstm_units, return_sequences=False, name='lstm_1')(inp)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation='relu', name='dense_shared')(x)
    x = Dropout(dropout_rate)(x)

    # Whites output: multi-label (sigmoid over 69)
    whites_out = Dense(MAX_WHITE, activation='sigmoid', name='whites_out')(x)

    # Red output: categorical (softmax over 26)
    red_out = Dense(MAX_RED, activation='softmax', name='red_out')(x)

    model = Model(inputs=inp, outputs=[whites_out, red_out], name='powerball_lstm')
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={'whites_out': 'binary_crossentropy', 'red_out': 'categorical_crossentropy'},
        loss_weights={'whites_out': 1.0, 'red_out': 1.0},
        metrics={'whites_out': ['accuracy'], 'red_out': ['accuracy']}
    )
    return model

# ---------------------------
# Training pipeline
# ---------------------------
def train(csv_path=CSV_PATH, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, epochs=EPOCHS, model_save_path='powerball_lstm.h5'):
    print("Loading data...")
    X, y_whites, y_red, draws, df = load_and_prepare(csv_path, seq_len=seq_len)
    print(f"Total training samples: {X.shape[0]} (seq_len={seq_len})")

    # Train/test split (time-aware)
    test_fraction = 0.2
    split_idx = int((1 - test_fraction) * X.shape[0])
    X_train, X_test = X[:split_idx], X[split_idx:]
    yw_train, yw_test = y_whites[:split_idx], y_whites[split_idx:]
    yr_train, yr_test = y_red[:split_idx], y_red[split_idx:]

    model = build_model(seq_len=seq_len)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        X_train,
        {'whites_out': yw_train, 'red_out': yr_train},
        validation_data=(X_test, {'whites_out': yw_test, 'red_out': yr_test}),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    # Save final
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    return model, history, (X, y_whites, y_red, draws, df)

# ---------------------------
# Prediction & sampling helpers
# ---------------------------
def predict_next_draw(model, recent_sequence, top_k_whites=10, pick_k=5, red_temperature=1.0, deterministic=True):
    """
    recent_sequence: np.array shape (seq_len, 95) - last seq_len draws encoded
    top_k_whites: consider top_k white candidates (from probs) and then sample pick_k unique ones
    pick_k: number of white balls to return (5)
    deterministic: if True, choose top pick_k; if False, sample among top_k using probabilities
    """
    if recent_sequence.ndim == 2:
        inp = recent_sequence.reshape(1, *recent_sequence.shape)
    else:
        inp = recent_sequence
    whites_probs, red_probs = model.predict(inp, verbose=0)
    whites_probs = whites_probs[0]  # length 69 (probabilities)
    red_probs = red_probs[0]        # length 26

    # Get candidate whites
    candidate_idx = np.argsort(whites_probs)[-top_k_whites:][::-1]  # indices (0-based)
    candidate_probs = whites_probs[candidate_idx]
    # Normalize candidate probs
    candidate_probs_norm = candidate_probs / (candidate_probs.sum() + 1e-12)

    if deterministic:
        chosen_indices = candidate_idx[:pick_k]
    else:
        # sample without replacement from candidate_idx using their normalized probabilities
        chosen_indices = np.random.choice(candidate_idx, size=pick_k, replace=False, p=candidate_probs_norm)

    whites = sorted([int(i + 1) for i in chosen_indices])  # convert to 1-indexed

    # Red: use temperature sampling
    # If red_probs are near-zero or uniform, decode_red handles log(small) safely
    red = decode_red_from_probs(red_probs, temperature=red_temperature)

    return whites, red, whites_probs, red_probs

# ---------------------------
# Historical verification
# ---------------------------
def check_combination_in_history(white_list, red_value, df):
    """
    df: original dataframe used (with White1..White5, Powerball columns)
    returns True + draw date if found, else False
    """
    white_sorted = sorted(white_list)
    for _, row in df.iterrows():
        draw_whites = sorted([int(row['White1']), int(row['White2']), int(row['White3']), int(row['White4']), int(row['White5'])])
        if draw_whites == white_sorted and int(row['Powerball']) == int(red_value):
            return True, row['DrawDate'] if 'DrawDate' in df.columns else None
    return False, None

# ---------------------------
# Example usage (if run as script)
# ---------------------------
if __name__ == "__main__":
    # 1) Train model (this will read CSV, preprocess, train)
    model, history, data_pack = train(csv_path=CSV_PATH, seq_len=SEQ_LEN, epochs=EPOCHS, model_save_path='powerball_lstm.h5')

    # 2) Load latest sequence and make predictions
    X_all, y_whites, y_red, draws, df = data_pack
    last_seq = X_all[-1]  # most recent sequence (seq_len, 95)
    # Generate multiple candidate tickets
    print("\nGenerated candidate tickets (deterministic top picks):")
    for i in range(5):
        whites, red, wp, rp = predict_next_draw(model, last_seq, top_k_whites=12, pick_k=5, red_temperature=0.8, deterministic=False)
        print(f"Ticket {i+1}: Whites: {whites}  Powerball: {red}")
        found, date = check_combination_in_history(whites, red, df)
        if found:
            print(f"   -> This exact combination appeared historically on: {date}")
        else:
            print("   -> Not found in history (based on the provided CSV)")

    # 3) If desired, save predictions / probabilities to disk for further analysis.

# Notes & recommendations
# 
# Data: Make sure powerball_history.csv is complete and clean. Official Powerball site or state lottery datasets provide authoritative histories. Columns must be DrawDate,White1..White5,Powerball. If your CSV uses different column names, tweak the code accordingly.
# 
# Targets and label order: The model predicts each white ball as multi-label across 69 outputs (sigmoid). During sampling we pick the top probabilities and ensure 5 unique whites. Order of whites in the actual draw is not important, so multi-label formulation is appropriate.
# 
# Model choices: LSTM is used because draws are a time sequence. You can experiment with GRU, stacked LSTMs, or Transformer encoders (for long-range dependency).
# 
# Evaluation: The per-output accuracy is misleading for multi-label tasks (baseline is low). Use metrics like mean average precision, recall@5, or custom hit-rate: “did the predicted top-5 include any true white?”.
# 
# Sampling: We provide both deterministic (top-k) and probabilistic (temperature) sampling. Temperature controls exploration in red sampling; you can also probabilistically sample whites rather than strictly top-k.
# 
# Compute: Training may be quick on small datasets but increase lstm_units/epochs for more capacity. Use GPU for faster training.