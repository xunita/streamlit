<h1 align="center">Data Wizard</h1><br>
<p align="center">
  <a href="https://thedatawizard.azurewebsites.net/" target="_blank">
    <img src="https://firebasestorage.googleapis.com/v0/b/hyphip-8ca89.appspot.com/o/datawiz.png?alt=media&token=5820f215-75f1-47ff-b486-b44d37aa02f7" alt="Data Wizard logo" height="140">
  </a>
</p>

<p align="center">
 DataWizard est une application web qui vous permet de visualiser vos données avec facilité.
</p>

## Démo

Consultez ce [lien](https://thedatawizard.azurewebsites.net/) pour une démo du site et de ses fonctionnalités.
## Prérequis

```bash
Python installé (version 3.12.1 utilisé pour ce projet)
```

## Environnement python

Vous devez créer un environnement python pour lancer le serveur

```bash
# Creation d'un environnement python
py -3 -m venv .venv 

# Activation de l'environnement
.venv\Scripts\activate (Microsoft Windows)

```

## Installation

Assurez vous d'installer les dépendances:

```bash
# Streamlit
pip install streamlit

# Seaborn
pip install seaborn

# Altair (si besoin)
pip install altair
```
## Serveur de développement

Démarrer le serveur sur `http://localhost:8501` (port par défault):

```bash
# Streamlit
streamlit run app.py
```

## Déploiment en ligne

Consultez cette [documentation de déploiement](https://docs.streamlit.io/knowledge-base/tutorials/deploy) pour plus d'informations.