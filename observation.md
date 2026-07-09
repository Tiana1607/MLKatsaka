# Analyse Comparative des Modèles — Diagnostic Rouille Polysora

## 1. Résultats obtenus

| Modèle | Accuracy | Précision | Recall |
|---|---|---|---|
| Arbre "from scratch" (Max-Minority) | 92.3% | 100% | 83.3% |
| Arbre Scikit-Learn (Gini) | 92.3% | 100% | 83.3% |
| Random Forest "from scratch" (Max-Minority) | 92.3% | 100% | 83.3% |
| Random Forest Scikit-Learn (Gini) | 84.6% | 100% | 66.7% |

---

## 2. Analyse du comportement

### Notre algorithme vs Scikit-Learn

Notre algorithme personnalisé basé sur la métrique **Max-Minority** obtient des performances **identiques** à celles de Scikit-Learn sur ce dataset (92.3% d'accuracy pour l'arbre, 92.3% pour notre Random Forest).

Ce résultat s'explique par la nature de nos données : la feature `pct_rouille` est suffisamment discriminante (corrélation de 0.62, seuil optimal à 1.66%) pour séparer les deux classes en un seul split. Dans ce cas, la métrique de pureté utilisée (Max-Minority ou Gini) importe peu — les deux convergent vers la même décision.

La métrique **Max-Minority** mesure la pureté de manière linéaire :

```
P(t) = max(proportion_classe_0, proportion_classe_1)
```

Tandis que **Gini** utilise une formule quadratique :

```
Gini(t) = 1 - Σ p²
```

Les deux métriques favorisent les nœuds purs, mais Max-Minority est plus simple et plus interprétable dans un contexte agronomique.

### Pourquoi la Forêt Aléatoire améliore la robustesse

Un arbre de décision unique présente un problème fondamental : il est **sensible aux variations du dataset**. Si on retire ou ajoute quelques images, l'arbre peut changer radicalement de structure.

La Forêt Aléatoire corrige ce problème via deux mécanismes :

**1. Le Bagging (Bootstrap Aggregating)**
Chaque arbre est entraîné sur un sous-échantillon tiré **avec remplacement**. Chaque arbre voit une version légèrement différente des données, ce qui introduit de la diversité entre les arbres.

**2. Le Vote Majoritaire**
La prédiction finale agrège les votes de tous les arbres. Une erreur isolée d'un arbre est "noyée" dans les votes des autres. Mathématiquement, si chaque arbre a une probabilité d'erreur ε < 0.5, la probabilité d'erreur de la forêt diminue exponentiellement avec le nombre d'arbres.

En résumé : **un arbre unique peut sur-apprendre**, une forêt **généralise mieux**.

> Dans notre cas, le Random Forest Scikit-Learn obtient un Recall plus faible (66.7% vs 83.3%) — ce qui s'explique par la petite taille du dataset (64 images). Avec davantage de données, le Random Forest surpasserait systématiquement l'arbre unique.

---

## 3. Discussion agronomique — Contexte malgache

### Les deux types d'erreurs

Dans le contexte agricole malgache, les deux types d'erreurs n'ont pas le même coût :

| Erreur | Description | Conséquence |
|---|---|---|
| **Faux Négatif** | Feuille malade classée saine | L'épidémie se propage → destruction des récoltes → insécurité alimentaire |
| **Faux Positif** | Feuille saine classée malade | Traitement fongicide inutile → coût financier pour l'agriculteur |

Dans les zones rurales malgaches où la culture du maïs est vitale pour la subsistance, un **Faux Négatif est bien plus coûteux** qu'un Faux Positif. Une épidémie de Rouille Polysora non détectée peut détruire une récolte entière, entraînant des pertes irréversibles pour des familles qui dépendent de cette culture.

### Quelle métrique privilégier ?

Dans ce contexte, nous devons **maximiser le Recall** (minimiser les Faux Négatifs) plutôt que la Précision.

- **Recall** = TP / (TP + FN) → mesure notre capacité à détecter toutes les feuilles malades
- **Précision** = TP / (TP + FP) → mesure notre capacité à éviter les fausses alarmes

### Modèle recommandé

Nous recommandons officiellement de déployer l'**Arbre de Décision "from scratch" (Max-Minority)** ou l'**Arbre Scikit-Learn (Gini)**, qui obtiennent tous deux :

- **Recall = 83.3%** → le meilleur parmi les quatre modèles
- **Précision = 100%** → aucun Faux Positif
- **Accuracy = 92.3%**

Le Random Forest Scikit-Learn, bien que plus sophistiqué, obtient un Recall de seulement 66.7% sur ce dataset — ce qui signifie qu'il rate davantage de feuilles malades, ce qui est inacceptable dans notre contexte.

### Recommandation finale

> Pour un déploiement sur le terrain auprès des techniciens agricoles malgaches, nous recommandons l'**Arbre de Décision avec la métrique Max-Minority**, pour les raisons suivantes :
> 1. **Recall maximal (83.3%)** — détecte le plus grand nombre de feuilles malades
> 2. **Précision parfaite (100%)** — ne génère aucune fausse alarme coûteuse
> 3. **Interprétabilité** — un technicien peut comprendre la règle de décision (`si pct_rouille > 1.66%` → malade)
> 4. **Légèreté** — fonctionne sur des appareils mobiles à faibles ressources, typiques en milieu rural malgache