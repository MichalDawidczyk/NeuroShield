from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from sklearn.model_selection import GridSearchCV

df_credit = pd.read_csv("datasets/creditcard.csv")

df_credit.fillna(df_credit.median(), inplace=True) # fill the missing values with median

df_credit.rename(columns={'Class': 'Fraud_Label'}, inplace=True)

scaler = MinMaxScaler()
df_credit[['Time', 'Amount']] = scaler.fit_transform(df_credit[['Time', 'Amount']]) # change the values to scale values, ML prefer scaled values

X = df_credit.drop(columns=['Fraud_Label'])
y = df_credit['Fraud_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train: {X_train.head()}")
print(f"X_test: {X_test.head()}")
print(f"y_train: {y_train.head()}")
print(f"y_test: {y_test.head()}")


xgb_model = XGBClassifier(eval_metric='logloss', n_estimators=200)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

with open("models/xgb_cassifier.pkl", "wb") as file:
    pickle.dump(xgb_model, file)

print("XGBClassifier model saved successfully!")