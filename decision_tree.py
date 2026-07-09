import numpy as np
import pandas as pd

def trouver_meilleur_split (X_column , y):
    indices_tri = np.argsort(X_column)
    x_trie = X_column[indices_tri]
    y_trie = y[indices_tri]
    meilleur_seuil = None
    purete_max = 0
    for i in range(len(x_trie) - 1):
        seuil = (x_trie[i] + x_trie[i+1]) / 2
        y_gauche = y_trie[x_trie <= seuil]
        y_droite = y_trie[x_trie > seuil]
        purete_gauche = purete(y_gauche)
        purete_droite = purete(y_droite)
        P_split = (len(y_gauche) * purete_gauche) / len(y) + (len(y_droite) * purete_droite) / len(y)
        if P_split > purete_max:
            purete_max = P_split
            meilleur_seuil = seuil
    return meilleur_seuil, purete_max


def purete(y):
    if len(y) == 0:
        return 0
    proportion_classe_1 = np.sum(y) / len(y)
    proportion_classe_0 = 1 - proportion_classe_1
    P = max(proportion_classe_1, proportion_classe_0)
    return P


def build_tree(X, y, depth, max_depth):
    if np.sum(y) == 0 or np.sum(y) == len(y) or depth >= max_depth : 
        return int(np.round(np.mean(y)))
    
    meilleure_purete = 0
    meilleur_seuil = None
    meilleure_feature = None

    for feature in range(X.shape[1]):
        seuil, purete = trouver_meilleur_split(X[:, feature], y)
        if purete > meilleure_purete:
            meilleure_purete = purete
            meilleur_seuil = seuil
            meilleure_feature = feature

        masque_gauche = X[:, meilleure_feature] <= meilleur_seuil
        X_gauche = X[masque_gauche]
        y_gauche = y[masque_gauche]
        X_droite = X[~masque_gauche]   
        y_droite = y[~masque_gauche]

        G = build_tree(X_gauche, y_gauche, depth + 1, max_depth)
        D = build_tree(X_droite, y_droite, depth + 1, max_depth)

    return {
        'feature': meilleure_feature,
        'seuil': meilleur_seuil,
        'gauche': G,
        'droite': D
    }


def predict(arbre, x):
    if isinstance(arbre, int):
        return arbre
    if x[arbre['feature']] <= arbre['seuil']:
        return predict(arbre['gauche'], x)
    else:
        return predict(arbre['droite'], x)

def predict_forest(arbres, x):
    votes = []   
    for arbre in arbres:
        votes.append(predict(arbre, x))
    return int(round(np.mean(votes)))


def random_forest(X, y, n_arbres, max_depth):
    arbres = []
    for i in range(n_arbres):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_echantillon = X[indices]
        y_echantillon = y[indices]

        arbre = build_tree(X_echantillon, y_echantillon, depth=0, max_depth=3)
        arbres.append(arbre)
    return arbres



df = pd.read_csv("features.csv")
X = df[['pct_rouille', 'rugosite', 'canny']].values
y = df['label'].values

arbre = build_tree(X, y, depth=0, max_depth=3)
print(arbre)