import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from decision_tree import build_tree, predict, random_forest, predict_forest
import pickle

df = pd.read_csv("features.csv")
X = df[['pct_rouille', 'rugosite', 'canny']].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

arbre_scratch = build_tree(X_train, y_train, depth=0, max_depth=3)
forest_scratch = random_forest(X_train, y_train, n_arbres=300, max_depth=3)


dt_sklearn = DecisionTreeClassifier(criterion='gini')
dt_sklearn.fit(X_train, y_train)

rf_sklearn = RandomForestClassifier(n_estimators=100)
rf_sklearn.fit(X_train, y_train)

y_pred_dt_sklearn = dt_sklearn.predict(X_test)
y_pred_rf_sklearn = rf_sklearn.predict(X_test)

y_pred_arbre_scratch = [predict(arbre_scratch, x) for x in X_test]
y_pred_forest_scratch = [predict_forest(forest_scratch, x) for x in X_test]
y_pred_dt = dt_sklearn.predict(X_test)
y_pred_rf = rf_sklearn.predict(X_test)

comparaison = {
    'modele' : ['dt_scratch', 'dt_sklearn', 'rf_scratch','rf_sklearn'],
    'accuracy' : [accuracy_score(y_test, y_pred_arbre_scratch), accuracy_score(y_test, y_pred_dt_sklearn), 
                  accuracy_score(y_test, y_pred_forest_scratch), accuracy_score(y_test, y_pred_rf_sklearn)],
    'precision' : [precision_score(y_test, y_pred_arbre_scratch), precision_score(y_test, y_pred_dt_sklearn), 
                  precision_score(y_test, y_pred_forest_scratch), precision_score(y_test, y_pred_rf_sklearn)],
    'recall' : [recall_score(y_test, y_pred_arbre_scratch), recall_score(y_test, y_pred_dt_sklearn), 
                  recall_score(y_test, y_pred_forest_scratch), recall_score(y_test, y_pred_rf_sklearn)]
}

dt = pd.DataFrame(comparaison)

print(dt)

# Sauvegarder
with open('modele.pkl', 'wb') as f:
    pickle.dump(forest_scratch, f)

# Charger
with open('modele.pkl', 'rb') as f:
    modele = pickle.load(f)