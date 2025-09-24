# Powerball ML Predictor (Educational Project)
# 
# This pipeline will:
# 
# Load historical Powerball data.
# 
# Prepare features (past draw numbers, frequencies, lag features).
# 
# Train a multi-label classification model (predicting the 5 white balls and the red ball separately).
# 
# Output “likely” combinations for the next draw.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import random

# ------------------------------
# 1. Load past Powerball data
# ------------------------------
# CSV columns: DrawDate, White1, White2, White3, White4, White5, Powerball
file_path = r'C:\Users\ankit\Desktop\JupyterProjects\Python\ML_prediction\powerball_history.csv'

df = pd.read_csv(file_path)

# Sort by date (oldest first)
df['DrawDate'] = pd.to_datetime(df['DrawDate'])
df = df.sort_values('DrawDate')

# ------------------------------
# 2. Feature Engineering
# ------------------------------
# Features: frequencies of each number up to that draw
max_white, max_red = 69, 26

# Initialize frequency trackers
white_freq = np.zeros(max_white+1)
red_freq = np.zeros(max_red+1)

features, targets = [], []

for _, row in df.iterrows():
    # Current feature = frequencies normalized
    feat = np.concatenate([
        white_freq[1:] / (white_freq.sum() + 1e-6),
        red_freq[1:] / (red_freq.sum() + 1e-6)
    ])
    features.append(feat)
    
    # Target = current draw (multi-label: 5 white balls + 1 red)
    targets.append([
        row['White1'], row['White2'], row['White3'], row['White4'], row['White5'], row['Powerball']
    ])
    
    # Update frequencies
    for n in [row['White1'], row['White2'], row['White3'], row['White4'], row['White5']]:
        white_freq[n] += 1
    red_freq[row['Powerball']] += 1

X = np.array(features)
y = np.array(targets)

# ------------------------------
# 3. Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ------------------------------
# 4. Multi-output Random Forest
# ------------------------------
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
clf.fit(X_train, y_train)

# ------------------------------
# 5. Evaluate
# ------------------------------
y_pred = clf.predict(X_test)

# Accuracy per ball (just for fun, real predictive power is tiny!)
for i, name in enumerate(["W1","W2","W3","W4","W5","PB"]):
    acc = accuracy_score(y_test[:,i], y_pred[:,i])
    print(f"{name} accuracy: {acc:.4f}")

# ------------------------------
# 6. Generate "Predicted" Next Draw
# ------------------------------
latest_features = X[-1].reshape(1, -1)
predicted = clf.predict(latest_features)[0]

print("\n🎰 Suggested ML-based numbers:")
print("White Balls:", sorted(predicted[:5]))
print("Powerball:", predicted[5])

#🔎 Explanation
#
#Features = past frequencies of numbers (white + red).
#
#Target = actual 5 white balls + 1 red ball.
#
#Model = Multi-output Random Forest (treats each ball as a separate classification).
#
#Output = "predicted" next draw numbers (really just frequency-driven guesses with some randomness).