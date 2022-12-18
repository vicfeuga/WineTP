# WineTP
By : Wassim Salablab, Baptiste Valentin, Victor Feuga

Veuillez trouver dans le notebook **analysis.ipynb** les démarches d'analyses des données pour mieux comprendre et maitriser les données.

Pour lancer le projet, il suffit d'entrer la commande *uvicorn main:app --reload*.
Puis taper sur les endpoints via Postman ou Firefox en inspectant le réseau et en éditant les requêtes

Pour nos prédictions
  - les vins ayant une qualité supérieur ou égale (>=) à 6 sont qualifiés comme **bons** (prédits par 1)
  - les vins ayant une qualité inférieur ou égale (<=) à 5 sont qualifiés comme **mauvais** (prédits par 0)
