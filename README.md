# Projet : Licence 3 Mathematiques - Techniques Statistiques

Efficacité d’une Centrale à Cycle Combiné à Gaz. 
Le but du projet était de réussir grâce au machine learning à concevoir un modèle prédictif pour une certaine variable de la base de données.
Pour ce projet les modèles de machine learning on été réalisé à partir de SAS. D'ou l'absence des modèles dans le code source.

Etape 1 : Préparation de la base de données (programmation R-Studio)
- Analyse statistique et descriptive (traitement univarié)
- Détection et suppression/gestion des valeurs manquantes et des valeurs aberrantes.
- Tests statistiques d'adéquation à une loi (Kolmogorov-Smirnov, Chi-deux, Shapiro-Wilk).
- Normalisation des données pour les tests et les modèles d'apprentissage.
- Visualisation des données et analyse poussée des corrélations entre variables. 

Etape 2 : Machine learning (R et SAS).
- Tests de plusieurs modèles de régression linéaire simple et multiple. Choix du meilleur modèle prédictif grâce au risque quadratique moyen de chaque modèle.
- Vérification des résultats obtenus à l'étape 1 grâce à une analyse en composante principale (ACP) .
- Vérification de l'existence potentielle de différentes classes dans notre base de données grâce à une Classification Ascendante Hiérarchique (CAH).
