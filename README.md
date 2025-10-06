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
- [Utilisation](#utilisation) : Cas d'utilisation du repo.
- [Amélioration](#amélioration) : Futur travail à réaliser.

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
b) **L'environnement de développement** : Les ressources physiques se limitent au capacités de calcul d'un ordinateur de bureau personnel (processeur AMD Ryzen 9900, 32Gb de RAM et GPU GeForce RTX 5060Ti). <br>
Voici un exemple de comment créer un environnement virtuel pour ce projet en utilisant `miniconda` (Linux) : 
```shell
conda create -n nba python=3.11
conda activate nba
pip install -r requirements.txt
```

Afin de pouvoir transférer de la donnée via des requête HTTPS (cf communication via l'API avec le serveur local), il est recommandé d'utiliser `curl`, qui peut être installé sous Linux via la commande : 

```shell
sudo apt install curl
```

Suivant cette logique, le contenu du rapport s'oriente vers une **PoC** du classifieur désiré. Les modèles étudiés tiennent compte de ces limitations.

## 3. Mise en production
**TODO** : DOC MISE EN PROD

# EDA
Le dataset étudié est **petit** (~1340 observations) et **bruité** (présence de redondances, de valeurs aberrantes, etc.). Les fichiers [analysis.ipynb](analysis.ipynb) et [pca.ipynb](pca.ipynb) contiennent une étude détaillée de la sélection des informations essentielles à l'entrainement des modèles. <br>
Les principales étapes sont : <br>
1. **Nettoyage des données** : Suppression des doublons, gestion des valeurs manquantes et correction des incohérences.
2. **Analyse univariée** : Étude des distributions de chaque variable (statistiques descriptives, histogrammes).
3. **Détection et traitement des outliers** : Identification des valeurs extrêmes et choix d'une stratégie (suppression, conservation ou capping). <br>
J'ai remarqué en particulier que l'utilisation de variables supplémentaires gardant trace des cappings effectués dégrade les performances des algorithmes (surtout ceux utilisant l'analyse de composantes principales, je pense que cela est dû au fait que ces variables catégoriques n'indiquent **pas quantitativement** l'écart entre la valeur tronquée et la valkeur limite, rendant difficile pour les modèles tels que la régression logistique de quantifier son impact sur la valeur cible).
4. **Analyse bivariée** : Exploration des relations entre variables (corrélations, visualisations).
5. **Sélection de variables** : Identification des variables pertinentes pour la modélisation.
6. **Préparation des jeux de données** : Création de versions filtrées et nivelées du dataset pour les étapes suivantes.

A l'issue de cette analyse, 2 nouvelles version du dataset donné sont produites : [nba_filtered.csv](src/data/nba_filtered.csv) qui contient les données filtrées avec **outliers** et [nba_filtered_capped.csv](src/data/nba_filtered_capped.csv) pour lequel les entrées correspondantes sont **nivelées** (ou limitées en valeur, utile pour les modèles sensibles aux outliers). <br>
Je choisis de laisser les labels de capping dans un dataset car cela pourrait être utile pour une évolution future du projet. De plus un exemple de code (lignes en commentaire) au sein de la classe `LREstimator` est laissé si besoin est d'utiliser cette extension.

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
Ces modèles sont parfois combinés avec une analyse de composantes principales afin de simplifier les données étudiées et améliorer (ou non) la performance des modèles en se focalisant sur l'information essentielle. <br>

# Sélection
Afin de choisir les **hyper-paramètres**, d'observer en détail les **performances** d'apprentissage de chaque modèle et de choisir le meilleur pour notre set de donnée, j'ai développé la stratégie suivante : <br>

0. **Reproducibilité** : une **seed** est fixée afin que chaque modèle travaille dans les mêmes conditions de calcul et reproduise les mêmes résultats sur chaque itération identique. Cela permet en particulier de comparer de manière équitable (mêmes conditions de calcul) les différents modèles utilisés. <br>
**Note** : Pour le code de régression logistique, `n_jobs` est fixé à 1 pour assurer la reproducibilité des résultats.
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
## 1. Train / Test
Le fichier [train](src/train.py) contient les fonctions nécessaires à l'entrainement du modèle choisi. <br>
Le fichier [test](src/test.py) teste en particulier la performance du modèle choisi. Je décide de changer la fonction de scoring pour y inclure le **score Fbeta** afin de rester cohérent avec l'analyse présentée ci-dessus. De plus je pense qu'afin de tester un modèle de manière plus équitable les sets d'entrainement et de test doivent posséder des **proportions équivalentes** vis à vis des **classes cibles**. <br>
Les deux fichiers ont besoin de connaitre certaines **configurations** vis à vis des modèles, localisées dans le fichier json [run_configs](src/run_configs.json). La partie "train" est à remplir manuellement, le script de training remplit automatiquement la partie "test". <br>

## 2. Conteneurisation


## 3. API
Le modèle entraîné est déployé dans un service web (local) à l'aide de la librairie `FastAPI` (format API REST). Le fichier [app](src/app.py) contient le détail du développement de l'API. Pour faire une requête sur un joueur, il suffit de lancer les commandes suivantes : 
```Shell
fastapi run src/app.py # launch API
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "gp": 0,
  "min": 0,
  "pts": 0,
  "fga": 0,
  "fg_percent": 0,
  "three_pa": 0,
  "three_p_percent": 0,
  "fta": 0,
  "ft_percent": 0,
  "oreb": 0,
  "reb": 0,
  "ast": 0,
  "stl": 0,
  "blk": 0,
  "tov": 0
}' # Remplacer les '0' par les valeurs souhaitées.
```

Voici la signification de chaque clé attendue dans la requête JSON pour la prédiction (Les valeurs sont en moyenne par match, à l'exception de `gp` qui comptabilise le nombre total de matchs joués) :

- **gp** : Nombre de matchs joués (Games Played)
- **min** : Minutes jouées au total
- **pts** : Points marqués au total
- **fga** : Nombre de tirs tentés (Field Goals Attempted)
- **fg_percent** : Pourcentage de réussite aux tirs (Field Goal %)
- **three_pa** : Nombre de tirs à 3 points tentés (Three Point Attempts)
- **three_p_percent** : Pourcentage de réussite à 3 points (Three Point %)
- **fta** : Nombre de lancers francs tentés (Free Throws Attempted)
- **ft_percent** : Pourcentage de réussite aux lancers francs (Free Throw %)
- **oreb** : Rebonds offensifs (Offensive Rebounds)
- **reb** : Rebonds totaux (Total Rebounds)
- **ast** : Passes décisives (Assists)
- **stl** : Interceptions (Steals)
- **blk** : Contres (Blocks)
- **tov** : Balles perdues (Turnovers)

# Utilisation
Le projet est prêt pour prouver son fonctionnement (modèle déjà choisi,entraîné, ...). <br>
Cette section détaille un cas d'utilisation de ce répertoire (pour le dataset fourni avec le sujet) si l'utilisateur souhaite rentrer plus en détail dans la construction du modèle: <br>
### 1. Analyse des données
Lancer le jupyter notebook [analysis.ipynb](src/analysis.ipynb) afin de traiter le dataset en entrée et générer les 2 nouveaux datasets associés (filtrés et/ou cappés).
### 2. Analyse des composantes principales
Lancer le jupyter notebook [pca.ipynb](src/pca.ipynb) afin d'analyser les variables à intégrer ou non dans la pipeline.
### 3. Fine-tuning
Les fichiers [knn.ipynb](src/knn.ipynb), [logreg.ipynb](src/logreg.ipynb), [svc.ipynb](src/svc.ipynb) et [random_forest.ipynb](src/random_forest.ipynb) sont à lancer afin de sélectionner les meilleur configurations et à comparer **ensuite** à l'aide du fichier [comparison.ipynb](src/comparison.ipynb).
### 4. Training / Testing
Exécuter le script [train.py](src/train.py) (penser à adapter le fichier [run_configs](src/run_configs.json) en fonction du modèle choisi) afin d'entraîner l'algorithm sélectionné. Le fichier [test.py](src/test.py) peut également être lancé pour en tester les performances.
### 5. Communication via l'API
Lancez l'API et envoyez votre requête comme présenté dans la [section précedente](#3-api).

# Amélioration
**TODO**  : présenter amélioration possibles