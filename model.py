from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("titanic")
df.head()

df.fillna(df.mean(numeric_only=True), inplace=True)

df.drop(columns=["embark_town", "class", "deck", "alive"], inplace=True)


df = pd.get_dummies(df, columns=["sex", "embarked", "who"], drop_first=True)

X = df[["age", "fare", "pclass", "alone",  "who_man", "who_woman", "embarked_Q", "embarked_S"]]

y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, None]
}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.score(X_test, y_test))
print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)
print("Test score:", grid.score(X_test, y_test))
dump(grid.best_estimator_, "model.pkl")
