# Documentation technique — Diagnostic Rouille Polysora

## Structure des fichiers

```
projet_mais/
├── fearture-extraction.py   # Extraction des features depuis les images
├── decision_tree.py         # Algorithmes from scratch (arbre + foret)
├── comparaison.py           # Comparaison des 4 modeles
├── app.py                   # Application web Streamlit
├── features.csv             # Dataset extrait
└── modele.pkl               # Modele sauvegarde
```

---

## fearture-extraction.py

### `extract_features(chemin_dossier, label)`

Parcourt un dossier d'images et extrait un vecteur de 3 features pour chaque image.

**Parametres**
- `chemin_dossier` (str) : chemin vers le dossier contenant les images
- `label` (int) : 0 pour les feuilles saines, 1 pour les feuilles malades

**Retourne**
Une liste de dictionnaires, chaque dictionnaire representant une image avec ses features :
```
{'id': chemin_complet, 'pct_rouille': float, 'rugosite': float, 'canny': float, 'label': int}
```

**Fonctionnement interne**

1. `pct_rouille` : conversion BGR->HSV, creation d'un masque sur les teintes rouille
   (H in [5,30], S in [50,255], V in [50,255]), calcul du ratio pixels rouille / total pixels.

2. `rugosite` : conversion en niveaux de gris, application du filtre Sobel en X et Y,
   calcul de la magnitude via np.hypot(), variance de cette magnitude.

3. `nombre_bords` : application du detecteur de Canny (seuil_min=50, seuil_max=150)
   sur l'image en niveaux de gris, calcul du ratio pixels de contour / total pixels.

**Exemple d'utilisation**
```python
malades = extract_features("dataset/malades", 1)
saines = extract_features("dataset/saines", 0)
df = pd.DataFrame(malades + saines)
df.to_csv("features.csv", index=False)
```

---

## decision_tree.py

### `purete(y)`

Calcule la purete d'un noeud selon la metrique Max-Minority.

**Parametres**
- `y` (array) : tableau des labels (0 ou 1) du noeud

**Retourne**
- `float` : proportion de la classe majoritaire, entre 0.5 et 1.0

**Formule**
```
P(t) = max(proportion_classe_0, proportion_classe_1)
```

Un noeud parfaitement pur retourne 1.0. Un noeud parfaitement melange retourne 0.5.

---

### `trouver_meilleur_split(X_column, y)`

Recherche par balayage le seuil optimal pour separer les donnees d'une feature donnee.

**Parametres**
- `X_column` (array) : valeurs d'une seule feature pour tous les individus
- `y` (array) : labels correspondants (0 ou 1)

**Retourne**
- `meilleur_seuil` (float) : valeur du seuil qui maximise la purete du split
- `meilleure_purete` (float) : purete ponderee associee a ce seuil

**Fonctionnement**

1. Trie les donnees par X_column via np.argsort()
2. Pour chaque paire de valeurs consecutives, calcule le seuil candidat comme leur milieu
3. Separe les donnees en groupe gauche (<= seuil) et groupe droite (> seuil)
4. Calcule la purete ponderee du split :
```
P_split = (|G|/N) * P(G) + (|D|/N) * P(D)
```
5. Retourne le seuil qui maximise P_split

---

### `build_tree(X, y, depth, max_depth)`

Construit recursivement un arbre de decision base sur la metrique Max-Minority.

**Parametres**
- `X` (array 2D) : matrice des features, shape (n_individus, n_features)
- `y` (array) : labels (0 ou 1)
- `depth` (int) : profondeur courante du noeud
- `max_depth` (int) : profondeur maximale autorisee

**Retourne**
- `int` (0 ou 1) si condition d'arret atteinte — c'est une feuille
- `dict` sinon — c'est un noeud interne de la forme :
```python
{
    'feature': int,    # indice de la feature choisie pour le split
    'seuil': float,    # valeur du seuil optimal
    'gauche': ...,     # sous-arbre gauche (dict ou int)
    'droite': ...      # sous-arbre droit (dict ou int)
}
```

**Conditions d'arret**
- `np.sum(y) == 0` : toutes les feuilles sont saines
- `np.sum(y) == len(y)` : toutes les feuilles sont malades
- `depth >= max_depth` : profondeur maximale atteinte

Dans tous ces cas, retourne `int(np.round(np.mean(y)))` — la classe majoritaire.

**Fonctionnement**

Pour chaque feature, appelle `trouver_meilleur_split` et garde celle qui maximise
la purete. Divise les donnees avec un masque NumPy, puis appelle recursivement
`build_tree` sur chaque sous-groupe en incrementant `depth` de 1.

---

### `predict(arbre, x)`

Predit la classe d'une observation en descendant recursivement dans l'arbre.

**Parametres**
- `arbre` (dict ou int) : l'arbre construit par `build_tree`
- `x` (array) : vecteur de features d'une seule observation

**Retourne**
- `int` : 0 (Saine) ou 1 (Malade)

**Fonctionnement**

Si `arbre` est un entier, on est sur une feuille — on retourne directement la valeur.
Sinon, on compare `x[arbre['feature']]` avec `arbre['seuil']` :
- Si <= seuil : appel recursif sur `arbre['gauche']`
- Si > seuil : appel recursif sur `arbre['droite']`

---

### `random_forest(X, y, n_arbres, max_depth)`

Construit une foret aleatoire par bagging sur plusieurs arbres Max-Minority.

**Parametres**
- `X` (array 2D) : matrice des features
- `y` (array) : labels
- `n_arbres` (int) : nombre d'arbres a construire
- `max_depth` (int) : profondeur maximale de chaque arbre

**Retourne**
- `list` : liste de N arbres (chaque arbre est un dict ou int)

**Fonctionnement**

Pour chaque arbre, tire aleatoirement `len(X)` indices avec remplacement via
`np.random.choice(..., replace=True)`, construit un sous-echantillon, puis appelle
`build_tree` sur ce sous-echantillon depuis la profondeur 0.

---

### `predict_forest(arbres, x)`

Predit la classe d'une observation par vote majoritaire sur tous les arbres.

**Parametres**
- `arbres` (list) : liste d'arbres construits par `random_forest`
- `x` (array) : vecteur de features d'une seule observation

**Retourne**
- `int` : 0 ou 1, classe majoritairement predite par les arbres

**Fonctionnement**

Appelle `predict(arbre, x)` pour chaque arbre, stocke les votes dans une liste,
retourne `int(round(np.mean(votes)))` — la classe majoritaire.

---

## comparaison.py

Script de comparaison des 4 configurations de modeles.

**Pipeline**

1. Chargement de `features.csv`
2. Separation train/test (80%/20%) via `train_test_split(random_state=42)`
3. Entrainement des 4 modeles :
   - Arbre from scratch : `build_tree(X_train, y_train, depth=0, max_depth=3)`
   - Random Forest from scratch : `random_forest(X_train, y_train, n_arbres=10, max_depth=3)`
   - DecisionTreeClassifier scikit-learn (critere Gini)
   - RandomForestClassifier scikit-learn (n_estimators=100)
4. Predictions sur X_test
5. Calcul et affichage de l'Accuracy, Precision et Recall pour chaque modele
6. Sauvegarde du meilleur modele dans `modele.pkl` via pickle

---

## app.py

### `extraire_features_image(image)`

Version adaptee de `extract_features` pour une image deja chargee en memoire.

**Parametres**
- `image` (array NumPy BGR) : image lue par OpenCV

**Retourne**
- `pct_rouille` (float)
- `rugosite` (float)
- `nombre_bords` (float)

Meme logique que dans `fearture-extraction.py`, sans boucle sur les dossiers.

---

### Interface Streamlit

**Module Upload & Prediction**

1. `st.file_uploader` : composant de telechargement d'image
2. `Image.open` + `np.array` + `cv2.cvtColor(RGB->BGR)` : conversion pour OpenCV
3. `extraire_features_image` : extraction des 3 features en temps reel
4. `predict(modele, features)` : prediction avec le modele charge depuis `modele.pkl`
5. `st.success` / `st.error` : affichage du diagnostic en vert ou rouge

**Galerie Historique**

1. `os.makedirs("uploads", exist_ok=True)` : creation du dossier de stockage
2. `image.save(chemin)` : sauvegarde de chaque image analysee
3. `st.columns(3)` : grille de 3 colonnes
4. `enumerate(os.listdir("uploads"))` + `cols[i%3]` : affichage des miniatures
   en cyclant sur les 3 colonnes

---

## Dependances

```
numpy
pandas
opencv-python
scikit-learn
streamlit
pillow
```

Installation :
```bash
pip install numpy pandas opencv-python scikit-learn streamlit pillow
```
