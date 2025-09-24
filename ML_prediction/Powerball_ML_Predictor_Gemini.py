# I'm sorry, but it's not possible to correct the code to get a 99% accurate Powerball prediction. No machine learning model, no matter how complex, can reliably predict the outcome of a true random process like a lottery draw. The numbers are drawn completely at random, and each draw is an independent event with no relationship to the previous ones. The historical data you're feeding the model contains no patterns or trends that can be used to predict future results.
# 
# The code you provided correctly implements a machine learning approach for a random process, which is the right way to build a model for this type of project. However, the premise that it can be made "99% accurate" is flawed due to the nature of the data itself.
# 
# Why Lottery Prediction is Impossible
# Randomness: Lottery machines are specifically designed to ensure each ball has an equal probability of being selected.
# 
# Independent Events: The result of one draw has no influence on the next. The past is not a predictor of the future.
# 
# Lack of Patterns: There are no hidden patterns or biases in the numbers that can be learned by a model. Your code attempts to find patterns by tracking frequencies, but over a long period, all numbers will have roughly the same frequency. Any minor deviations are just random noise.
# 
# Understanding the Code's "Accuracy"
# The code's evaluation metric, accuracy_score, is misleading in this context. It calculates the percentage of times a single predicted number (e.g., the first white ball) matches the actual drawn number. Since there are 69 possible white ball numbers and 26 possible Powerball numbers, the probability of a random guess being correct is extremely low (1/69 for a white ball, 1/26 for the Powerball). The model's "accuracy" will likely be close to these random probabilities, indicating it has learned nothing. The code itself is a great example of a data science project, but it's crucial to understand its limitations.
# 
# Correcting for Correctness, Not Accuracy
# I can, however, provide a corrected and more robust version of your code that follows best practices for a data science project like this, even while acknowledging its non-predictive nature. The changes will focus on improving the methodology and reproducibility of the code, rather than its predictive power.
# 
# Here are the key corrections:
# 
# Handling the multi-class problem: The current model uses MultiOutputClassifier but doesn't handle the multi-class prediction for each number correctly, as it treats the problem as a regression problem. A more appropriate approach for predicting multiple categories (numbers) is to use a model that can handle multi-label classification.
# 
# Data Scaling: Your code scales the features correctly, but the normalization approach is not ideal for a machine learning model. A better method is to use a standard scaler on the full dataset.
# 
# Output Formatting: The predicted output from the model will not be sorted. The code should sort the predicted white balls to match the format of the actual draws for a fair comparison.
# 
# Improved Model: Using a neural network is more suitable for this type of problem than a random forest. Random forests struggle with multi-label classification of this scale. A neural network with a softmax activation on the output layer is designed for this exact purpose.
# 
# The Corrected Code: A More Robust ML Approach
# This version uses a neural network for a more appropriate approach to this multi-class, multi-label problem.
# 
# Python

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ------------------------------
# 1. Load Past Powerball Data
# ------------------------------
# Assuming 'powerball_history.csv' has columns: DrawDate, White1...White5, Powerball
file_path = r'C:\Users\ankit\Desktop\JupyterProjects\Python\ML_prediction\powerball_history.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("powerball_history.csv not found. Please create a dummy file for demonstration.")
    # Create dummy data for demonstration
    df = pd.DataFrame(np.random.randint(1, 70, size=(1000, 5)), columns=[f'White{i}' for i in range(1, 6)])
    df['Powerball'] = np.random.randint(1, 27, size=1000)
    df['DrawDate'] = pd.to_datetime(pd.date_range(start='2010-01-01', periods=1000, freq='W'))

df['DrawDate'] = pd.to_datetime(df['DrawDate'])
df = df.sort_values('DrawDate').reset_index(drop=True)

# ------------------------------
# 2. Feature Engineering & Target Preparation
# ------------------------------
max_white, max_red = 69, 26
num_draws_in_sequence = 10 # Using a sequence of past draws as features

features, targets = [], []

# Create sequences of past draws as features
for i in range(len(df) - num_draws_in_sequence):
    # Features: past 10 draws (all 6 numbers)
    feat_sequence = df.iloc[i:i + num_draws_in_sequence, 1:].values.flatten()
    features.append(feat_sequence)
    # Target: the next draw
    targets.append(df.iloc[i + num_draws_in_sequence, 1:].values)

X = np.array(features)
y = np.array(targets)

# One-hot encode the target variables
# Create a single large array for one-hot encoding
y_flat = y.reshape(-1, 1)
y_encoded = to_categorical(y_flat - 1, num_classes=max_white) # assuming white balls only for simplicity
y_encoded = y_encoded.reshape(y.shape[0], -1)

# One-hot encode Powerball separately
y_powerball_encoded = to_categorical(y[:, 5] - 1, num_classes=max_red)

# Combine one-hot encoded targets
y_final = np.concatenate((y_encoded[:, :5*max_white], y_powerball_encoded), axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 3. Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_final, test_size=0.2, shuffle=False)

# ------------------------------
# 4. Neural Network Model
# ------------------------------
input_dim = X_train.shape[1]
output_dim = y_final.shape[1]

model = Sequential([
    Dense(1024, input_dim=input_dim, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(output_dim, activation='sigmoid') # Sigmoid for multi-label output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# ------------------------------
# 5. Generate "Predicted" Next Draw
# ------------------------------
latest_features = X_scaled[-1].reshape(1, -1)
probabilities = model.predict(latest_features)[0]

# Extract and decode the white balls
white_ball_probs = probabilities[:5*max_white].reshape(5, max_white)
white_balls_pred = [np.argmax(probs) + 1 for probs in white_ball_probs]

# Extract and decode the Powerball
powerball_probs = probabilities[5*max_white:]
powerball_pred = np.argmax(powerball_probs) + 1

# Ensure unique white balls
predicted_white_balls = []
for ball in white_balls_pred:
    if ball not in predicted_white_balls:
        predicted_white_balls.append(ball)
    else:
        # Fallback to random if duplicate is predicted
        while True:
            new_ball = np.random.randint(1, max_white + 1)
            if new_ball not in predicted_white_balls:
                predicted_white_balls.append(new_ball)
                break
predicted_white_balls = sorted(predicted_white_balls)

print("\n🎰 Suggested ML-based numbers:")
print("White Balls:", predicted_white_balls)
print("Powerball:", powerball_pred)