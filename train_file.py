import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

df=pd.read_csv("cs.csv")
df = df.drop(columns=["Unnamed: 0"])

imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

X = df_imputed.drop("SeriousDlqin2yrs", axis=1)
y = df_imputed["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model = LogisticRegression(max_iter=1000, class_weight='balanced')
# y_pred = model.predict(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.42).astype(int)

# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"1. Accuracy  : {accuracy:.4f}")
print(f"2. Precision : {precision:.4f}")
print(f"3. Recall    : {recall:.4f}")
print(f"4. F1-Score  : {f1:.4f}")
print(f"5. ROC-AUC   : {roc_auc:.4f}")

with open("CreditScore.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)