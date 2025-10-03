# MP DATA INTERVIEW
Voici mon rapport sur le Test technique Data Scientist Junior chez MP Data. <br>

# Intro
Ce répertoire détaille mon analyse complète sur le sujet suivant : développer et déployer un **classifieur** afin de prédire si un jeune athlète de NBA va voir se carrière durer plus de 5 ans au vu de ces statistiques sportives. <br>

Mon rendu s'organise de la façon suivante : 
## Table des matières

- [Objectifs](#objectifs) : Définition des objectifs business.
- [EDA](#eda) : Exploration des données.
- [Modèle](#modèle) : Développement de potentiels candidats.
- [Sélection](#sélection) : Comparaison et choix du modèle à mettre en production.
- [Déploiement](#web) : Déploiement du modèle dans un service web.

# Objectifs
Après analyse du sujet et discussion avec l'équipe de recrutement, j'ai pu identifier les axes pilliers de l'élaboration d'un bon projet data pour répondre à la problématique posée, amenant aux configurations suivantes: <br>
## 1. Besoins des investisseurs <br>
Mon analyse simule une réponse à des clients qui considèrent qu'il est **2 fois pire d'investir dans un mauvais athlète que d'en rater un bon** (carrière future dure plus de 5 ans). J'ai donc pour cela modifié la fonction de scoring proposée (rappel) pour $F_\beta$ définie comme :

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Précision} \times \text{Rappel}}{(\beta^2 \times \text{Précision}) + \text{Rappel}}
$$

où $\beta =0.5 < 1$, donnant 2 fois d'importance à la précision qu'au rappel.

## 2. Environnement de développement
Mon analyse simule également les 2 contraintes suivantes : <br>
a) **Les données sources** : Les seules données accessibles sont les données présentes dans le fichier csv associé au test. Je suppose que celles ci ont été extraites sur ce [site](https://www.basketball-reference.com/). <br>
b) **L'environnement de développement** : Les ressources physiques se limitent au capacités de calcul d'un ordinateur de bureau personnel (le mien en l'occurence). <br>

Suivant cette logique, le contenu du rapport s'oriente vers une **PoC** sur le classifieur désiré. Les modèles étudiés tiennent compte de ces limitations.

## 3. Mise en production
**TODO** : DOC MISE EN PROD

# EDA
Le dataset étudié est **petit** (~1340 observations) et **bruité** (présence de redondances, de valeurs aberrantes, etc.). Les fichiers [analysis.ipynb](analysis.ipynb) et [pca.ipynb](pca.ipynb) contiennent une étude détaillée de la sélection des informations essentielles à l'entrainement des modèles. <br>
Les principales étapes sont : <br>
1. **Nettoyage des données** : Suppression des doublons, gestion des valeurs manquantes et correction des incohérences.
2. **Analyse univariée** : Étude des distributions de chaque variable (statistiques descriptives, histogrammes).
3. **Détection et traitement des outliers** : Identification des valeurs extrêmes et choix d'une stratégie (suppression, conservation ou capping).
4. **Analyse bivariée** : Exploration des relations entre variables (corrélations, visualisations).
5. **Sélection de variables** : Identification des variables pertinentes pour la modélisation.
6. **Préparation des jeux de données** : Création de versions filtrées et nivelées du dataset pour les étapes suivantes.

A l'issue de cette analyse, 2 nouvelles version du dataset donné sont produites : [nba_filtered.csv](src/data/nba_filtered.csv) qui contient les données filtrées avec **outliers** et [nba_filtered_capped.csv](src/data/nba_filtered_capped.csv) pour lequel les entrées correspondantes sont **nivelées** (ou limitées en valeur, utile pour les modèles sensibles aux outliers).

# Modèle
Une fois les données nettoyées, j'ai réfléchi à quelques modèles qui pourraient bien être adapté à ces dernières. Il faut prendre en compte que le **dataset est petit** et que certaines **variables restent très corrélées** (sans pour autant introduire de la redondance).
Voici une table récapitulative des modèles choisis : 

| Modèle                      | Avantages                                                                 | Inconvénients                                                        | Complexité entraînement / test | Complexité mémoire         |
|-----------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------|-------------------------------|---------------------------|
| [Logistic Regression](src/logreg.ipynb)        | Simple, interprétable, rapide, régularisation possible pour éviter le surapprentissage          | Suppose linéarité entre les features, peu performant si relations complexes             | $O(n\times d)$ / $O(d)$                  | $O(d)$                      |
| [Random Forest](src/random_forest.ipynb)               | Gère non-linéarités, robuste aux outliers, peu de tuning nécessaire      | Moins interprétable, plus lent à entraîner      | $O(n_{trees}\times n\times \log n)$ / $O(n_{trees})$ | $O(n_{trees}\times n\times d)$             |
| [Support Vector Machine](src/svc.ipynb)          | Performant sur petits datasets, efficace pour des séparations complexes           | Sensible au choix des paramètres (le noyau notamment)       | $O(n^3)$ (noyau linéaire)           | $O(n^2)$                 |
| [K-Nearest Neighbors](src/knn.ipynb)   | Simple, sans entraînement, non paramétrique                              | Lenteur au test, sensible au bruit et à la dimension                 | $O(1) / O(n\times d)$                | $O(n\times d)$                    |

Avec $n$ le nombre d'**observations** dans le set d'entraînement, $d$ le nombre de **features** et $n_{trees}$ le nombre d'**arbres**. <br>
Chaque est associé à un notebook détaillant son analyse. Le répertoire [models](src/models/) contient une implémentation détaillées de chaque modèle (ce qui permet plus de flexibilité vis à vis des fonctions à développer). <br>
Ces modèles sont parfois combinés avec une analyse de composantes principales afin de simplifier les données étudiées et améliorer (ou non) la performance des modèles en se focalisant sur l'information essentielle.

# Sélection
Afin de choisir les **hyper-paramètres**, d'observer en détail les **performances** d'apprentissage de chaque modèle et de choisir le meilleur pour notre set de donnée, j'ai développé la stratégie suivante : <br>

0. **Reproducibilité** : une **seed** est fixée afin que chaque modèle travaille dans les mêmes conditions de calcul et reproduise les mêmes résultats sur chaque itération identique.
1. **Séparation** : 80% du set initial est utilisé pour entraîner les modèles et 20% pour le test final. Les splits sont **stratifiés** afin de tenir compte de la différence en proportion des classes cibles. Lors de la **validation croisée**, j'assure que la taille des sets de validation est la même que celle du test (assure une bonne généralisation des résultats).
2. **Fine-tuning** : Chaque classe (associée à un modèle) implémente une méthode d'entrainement et une méthode de validation croisée qui retourne les scores associés (permet de voir la statibilité du modèle à travers différents scénarios). <br>
De plus pour les modèles nécéssitants du fine-tuning d'hyper-paramètres, j'ai choisi le framework [OPTUNA](https://optuna.org/), qui implémente un algorithme d'**optimisation Bayésienne**. Cela permet de définir des intervales spécifiques (choisi empiriquement) pour adapter de manière efficace et crédible (intervalles de confiance basé sur des tests d'hyphotèses statistiques) les modèles concernés. <br>
Les meilleures configurations associées à chaque modèle sont sauvegardées dans ce [répertoire](src/models/params/).
3. **Visualisation** : Au sein des notebooks, de nombreuses visualisations en détail sont effectuées pour choisir le modèle qui offre le meilleur **ratio performance / stabilité**. Chaque modèle est au moins comparé à l'algorithme stupide qui prédit la classe cible qu'il a le plus observé.

#### Conclusion
De tous les algortihmes étudiés, celui qui retient le plus mon attention est **PCA + Régression logistique + Régularisation L2**, voici pourquoi : <br>
1) Le modèles est **simple** (rapide, peu d' hyper-paramètres) à entraîner / fine-tuner et occupe peu de place en mémoire comparer aux autres algorithmes présentés ci dessus.
2) La **PCA** permet de se focaliser sur l'information essentielle contenue dans le dataset et gère la forte corrélation entre certaines features.
3) La **régularization L2** est un moyen de prévenir un surapprentissage sur les données d'entrainement, rendant le modèle résilient.
4) Les courbes d'apprentisage sont relativement **stables** à travers les différents sets d' entrainement / de validation et de test (surtout pour la précision). De plus l'algorithme se généralise très bien au données non encore observées (et qui possèdent les mêmes proportions en classes cibles que le set d'entrainement).
5) Cet algorithme présente les meilleurs résultats (score fbeta) en validation par rapport aux autres modèles (voir [comparaison](src/comparison.ipynb))

# Déploiement
Le fichier [train](src/train.py) contient les fonctions nécessaires à l'entrainement du modèle choisi. <br>
Le fichier [test](src/test.py) teste en particulier la performance du modèle choisi. Je décide de changer la fonction de scoring pour y inclure le **score Fbeta** afin de rester cohérent avec l'analyse présentée ci-dessus. De plus je pense qu'afin de tester un modèle de manière plus équitable les sets d'entrainement et de test doivent posséder des **proportions équivalentes** vis à vis des **classes cibles**. <br>
Les deux fichiers ont besoin de connaitre certaines **configurations** vis à vis des modèles, localisées dans le fichier json [run_configs](src/run_configs.json). La partie "train" est à remplir manuellement, le script de training remplit automatiquement la partie "test".


# Amélioration
**TODO**  : présenter amélioration possibles