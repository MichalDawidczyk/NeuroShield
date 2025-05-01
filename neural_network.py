import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df_credit = pd.read_csv("datasets/creditcard.csv")

df_credit.fillna(df_credit.median(), inplace=True) # fill the missing values with median

df_credit.rename(columns={'Class': 'Fraud_Label'}, inplace=True)

scaler = StandardScaler()
df_credit[['Time', 'Amount']] = scaler.fit_transform(df_credit[['Time', 'Amount']]) # change the values to scale values, ML prefer scaled values

X = df_credit.drop(columns=['Fraud_Label'])
y = df_credit['Fraud_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

print(f"X_train: {X_train.head()}")
print(f"X_test: {X_test.head()}")
print(f"y_train: {y_train.head()}")
print(f"y_test: {y_test.head()}")

nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='nadam', loss="binary_crossentropy", metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
print("Total Fraud Cases:", sum(y_train))
print("Total Non-Fraud Cases:", len(y_train) - sum(y_train))
print("Example Fraud Labels:", y_train)
loss, accuracy = nn_model.evaluate(X_test, y_test)
print(f"Neural Network Accuracy: {accuracy:.6f}")

nn_model.save("models/neural_network.keras")

print("Neural Network model saved successfully!")
