import sys
from datetime import datetime
import re

# importation des librairies pour le traitement des données

import streamlit as st # librairie de streamlit

# librairie de visualisation de données
import seaborn as sns 
import matplotlib.pyplot as plt

# librairie de traitement de données
import pandas as pd
import csv

# st.set_option('server.enableCORS', True)

# J'ai ajouté cette ligne pour éviter un warning de Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

# Permet de changer le titre de la page, l'icône et la mise en page
st.set_page_config(
    page_title="DataWiz | DataWiz est une application web qui permet de visualiser et de traiter des données.",  
    page_icon="https://firebasestorage.googleapis.com/v0/b/hyphip-8ca89.appspot.com/o/datawiz.png?alt=media&token=5820f215-75f1-47ff-b486-b44d37aa02f7", 
    layout="wide",   # Permet d'afficher d'utiliser la totalité de la page
    initial_sidebar_state="expanded",  # Permet d'afficher la sidebar par défaut
)

# lien vers le logo de la page
logo = "https://firebasestorage.googleapis.com/v0/b/hyphip-8ca89.appspot.com/o/datawiz.png?alt=media&token=5820f215-75f1-47ff-b486-b44d37aa02f7"

# fonction pour ordonner les données selon une colonne et un ordre
def order_by(dataframe, column, ascending=True):
  
  datacopy = dataframe.copy()
  return datacopy.sort_values(by=column, ascending=ascending)

# fonction pour détecter le séparateur d'un fichier csv
def detect_separator(uploaded_file):
  # Convertir le fichier en string
  content = uploaded_file.getvalue().decode('utf-8')

  # Utiliser Sniffer pour détecter le séparateur
  # 8192 pour traiter 8kb(premiers caractères) de données mais on peut
  # ajuster la valeur selon la taille du fichier max prévu pour l'application.
  # Augmenter jusqu'à fonctionnement si le fichier est plus grand.
  dialect = csv.Sniffer().sniff(content[:8192])

  return dialect.delimiter

# Affichage de la matrice de corrélation
def correlation_matrix(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Correlation Matrix")
  corr = dataframe[numeric_columns].corr()
  sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
  st.pyplot(fig)

# Affichage du graphique de comptage
def count_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Count Plot")
  x_column = st.selectbox("Select column Count Plot:", dataframe.columns)
  sns.countplot(data=dataframe[x_column], ax=ax)
  st.pyplot(fig)
  
# Affichage du graphique en secteur
def pie_chart(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Pie Chart")
  x_column = st.selectbox("Select column Pie Chart:", dataframe.columns)
  dataframe[x_column].value_counts().plot(kind='pie', ax=ax)
  st.pyplot(fig)
  
# Affichage du graphique en barres
def bar_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Bar Plot")
  x_column = st.selectbox("Select X-axis column Bar Plot:", dataframe.columns)
  if(len(dataframe.columns) > 1):
    y_column = st.selectbox("Select Y-axis column Bar Plot:", dataframe.columns)
  sns.barplot(x=x_column, y=y_column, data=dataframe, ax=ax)
  st.pyplot(fig)
  
# Affichage du graphique en nuage de points
def scatter_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Scatter plot")
  x_axis = st.selectbox("Select X-axis for Scatter:", dataframe.columns)
  y_axis = st.selectbox("Select Y-axis for Scatter:", dataframe.columns)
  sns.scatterplot(x=dataframe[x_axis], y=dataframe[y_axis], ax=ax)
  ax.set_title("Scatter Plot")
  ax.set_xlabel(x_axis)
  ax.set_ylabel(y_axis)
  st.pyplot(fig)
  
# Affichage du graphique en ligne
def line_chart(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Line Chart")
  x_column = st.selectbox("Select X-axis column Line Chart:", numeric_columns)
  y_column = st.selectbox("Select Y-axis column Line Chart:", numeric_columns)
  line_chart_data = dataframe[[x_column, y_column]]
  ax.plot(line_chart_data[x_column], line_chart_data[y_column])
  ax.set_xlabel(x_column)
  ax.set_ylabel(y_column)
  st.pyplot(fig)

# Affichage du graphique en pair
def pair_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader(f"Pair Plot")
  x_column = st.selectbox("Select X-axis column:", dataframe.columns)
  y_column = st.selectbox("Select Y-axis column:", dataframe.columns)
  st.write(f"Customized X: {x_column}, Y: {y_column}")
  sns.pairplot(dataframe, x_vars=[x_column], y_vars=[y_column], height=7)
  ax.set_xlabel(x_column)
  ax.set_ylabel(y_column)
  st.pyplot()

# Affichage du graphique en histogramme
def histogram(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Histogram")
  column_name = st.selectbox("Select a column:", dataframe.columns)
  # Plot histogram with vertical orientation for categorical variables
  if dataframe[column_name].dtype == 'category':  # Check if the column is categorical
    ax.hist(dataframe[column_name], orientation='horizontal', bins='auto')
  else:
    ax.hist(dataframe[column_name])
  ax.set_xlabel(column_name)
  st.pyplot(fig)
  
# Affichage du graphique en boîte
def box_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Box Plot")
  x_column = st.selectbox("Select X-axis column for Box Plot:", dataframe.columns)
  y_column = st.selectbox("Select Y-axis column for Box Plot:", dataframe.columns)
  st.write(f"Customized X: {x_column}, Y: {y_column}")
  sns.boxplot(x=x_column, y=y_column, data=dataframe, ax=ax)
  ax.set_xlabel(x_column)
  ax.set_ylabel(y_column)
  st.pyplot(fig)
  
# Affichage du logo et
st.markdown(f"""
    <div style="display: flex; align-items: center;">
    <a href="/">
        <img style="width: 80px; height: 76px; border-radius: 64px;" src="{logo}" alt="Logo" width="300">
    </a>
    <h1 style="text-align: center; font-size: 48px; font-weight: bold;">DataWiz</h1></div>
""", unsafe_allow_html=True)


# Chargement du fichier (csv seulement pour le moment, à améliorer si intéressé(e))
uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")

# Si un fichier est chargé (taille <= 200 MB) alors on continue
if uploaded_file is not None:
    # Convertir le fichier en bytes
    bytes_data = uploaded_file.getvalue()
    try:
      # Détecter le séparateur du fichier
      separator = detect_separator(uploaded_file)
      # Convertir le fichier en dataframe
      dataframe = pd.read_csv(uploaded_file, header=[0], sep=separator)
      
    except:
      # Afficher un message d'erreur si le séparateur n'est pas détecté
      st.error("Unable to detect separator. Please check your file and try again.")
      # Arrêter l'exécution du script
      sys.exit()
      
    # Nom du site
    st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center;">
    <a href="/">
        <img style="width: 60px; height: 56px; border-radius: 64px;" src="{logo}" alt="Logo" width="300">
    </a>
    <h1 style="text-align: center; font-size: 38px; font-weight: bold; margin-left:10px;">DataWiz</h1></div>
""", unsafe_allow_html=True)
    
    # Titre de la sidebar
    st.sidebar.header("Search & metadata")
    
    # Copie de la base de donnée
    copy = dataframe.copy()

    # Récuperation de la taille de la base de donnée
    taille= len(copy.columns)
    
    # Création des filtres
    filtre_par_colonne= st.sidebar.selectbox("Select a feature", copy.columns[0:taille])
    
    def is_potential_date(column):
      # Define regex patterns for both French and US date formats
      p1= re.compile(r"\d{1,2}[.-]\d{1,2}[.-]\d{4}( \d{2}:\d{2}:\d{2})?")
      p2 = re.compile(r"\d{4}[.-]\d{1,2}[.-]\d{1,2}( \d{2}:\d{2}:\d{2})?")
  
      # Check if at least one value in the column matches either date pattern
      if column.dtype == 'O' or column.dtype == 'object' or column.dtype == 'category':
        return any(column.str.match(p1, na=False)) or any(column.str.match(p2, na=False))
      return False
    # Variable de detection de type de colonne
    colType = "string/number"
    
    if pd.api.types.is_bool_dtype(copy[filtre_par_colonne]):
      colType = "boolean"
      filtre_search_boolean= st.sidebar.selectbox("Select a boolean value ",("True"  ,"False"))
    
    elif pd.api.types.is_datetime64_any_dtype(copy[filtre_par_colonne]):
        colType = "datetime"
        start_date = datetime(1976, 1, 1)
        end_date = datetime.today()
        selected_range = st.sidebar.date_input("Select a date range", 
                                  min_value=start_date, 
                                  max_value=end_date, 
                                  value=(start_date, end_date))
    else:
        colType = "string/number"
        if(is_potential_date(copy[filtre_par_colonne])):
          st.sidebar.write("This column seems to contain dates. Please select a date range.")
          colType = "datetime"
          start_date = datetime(1976, 1, 1)
          end_date = datetime.today()
          selected_range = st.sidebar.date_input("Select a date range", 
                                  min_value=start_date, 
                                  max_value=end_date, 
                                  value=(start_date, end_date))
        else:
          filtre_search_word= st.sidebar.text_input("Insert a " + "'"+filtre_par_colonne+"'" " value for search ", value=None, placeholder="Type a value...")
    
    filtre_ascending_des= st.sidebar.selectbox("Order by ",("Ascending"  ,"Descending"))
    filtre_null= st.sidebar.selectbox("Include null values",("Yes"  ,"No"))
    
    #True si le filtre null est activé et False sinon
    is_accessible = True

    # Affiche ou non les valeurs nulles selon le filtre null
    if str(filtre_null)=='No' :
      is_accessible = False
      copy = copy.copy().dropna()
    else :
      is_accessible = True
      copy = copy

    # Placé ici en raison de la dépendance de la variable is_accessible
    rows_with_null = st.sidebar.checkbox('Show rows with only null values', value=False, key="accessible_checkbox_key", help="This checkbox can be toggled based on 'Include null values' state.", disabled=not is_accessible)

    filtre_null_stat= st.sidebar.selectbox("Null Values Statistics",copy.columns[0:taille])

    if colType == "string/number":
        # Traitements des données selon les filtres de recherche par mot clé et par colonne
        if filtre_search_word:
            # Vérifier si la colonne est de type string
            if copy[filtre_par_colonne].dtype == 'O' or copy[filtre_par_colonne].dtype == 'object' or copy[filtre_par_colonne].dtype == 'category':
                copy = copy[copy[filtre_par_colonne].astype(str).str.contains(filtre_search_word, case=False, na=False)]
            else:
                try:
                    # Essayer de convertir le mot clé en type de la colonne
                    converted_search_term = type(copy[filtre_par_colonne].iloc[0])(filtre_search_word)
                    copy = copy[copy[filtre_par_colonne] == converted_search_term]
                except (ValueError, TypeError):
                    # Afficher un message d'erreur si le mot clé ne peut pas être convertir
                    st.sidebar.error("Unable to convert search term to string/numeric.")
                    pass
                  
    elif colType == "boolean":
      if filtre_search_boolean:
        try:
          
          if filtre_search_boolean == "True":
            copy = copy[copy[filtre_par_colonne] == True]
          else:
            copy = copy[copy[filtre_par_colonne] == False]
        except (ValueError, TypeError):
            # Afficher un message d'erreur si le mot clé ne peut pas être convertir
          st.sidebar.error("Unable to convert search term to boolean.")
          pass
        
    elif colType == "datetime":
      if selected_range:
        try:
          # Vérifie si seconde date est sélectionnée avant de filtrer
          if len(selected_range ) > 1:
            copy = copy[(pd.to_datetime(copy[filtre_par_colonne], format="ISO8601") >= pd.to_datetime(selected_range[0], format="ISO8601")) & (pd.to_datetime(copy[filtre_par_colonne], format="ISO8601") <= pd.to_datetime(selected_range[1], format="ISO8601"))]
          else:
            copy = copy[pd.to_datetime(copy[filtre_par_colonne], format="ISO8601") >= pd.to_datetime(selected_range[0], format="ISO8601")]
        except (ValueError, TypeError):
          st.sidebar.error("Unable to convert search term to date.")
          pass
        
    # Traiter les données pour afficher les lignes avec des valeurs nulles
    if rows_with_null:
      copy = copy[copy.isnull().any(axis=1)]

    # Affiche le pourcentage de valeurs nulles dans la colonne sélectionnée
    null_stats = copy[str(filtre_null_stat)].isnull().sum()
    total_rows = len(copy)
    null_percentage = (null_stats / total_rows) * 100
    if total_rows == 0:
      st.sidebar.write(f"{str(filtre_null_stat)}: {null_stats}/{total_rows} row(s), about 0.00%")
    else:
      st.sidebar.write(f"{str(filtre_null_stat)}: {null_stats}/{total_rows} row(s), about {null_percentage:.2f}%")

    # Appliquer le filtre d'ordre
    if str(filtre_ascending_des)=='Ascending' :
      copy = order_by(copy, filtre_par_colonne, ascending=True)
    else :
      copy = order_by(copy, filtre_par_colonne, ascending=False)

    # Copie de la base de donnée après les traitements
    after_filtre=copy.copy()
    
    # Affichage de la base de donnée
    st.dataframe(after_filtre, use_container_width=True, hide_index=True)
    num_rows = after_filtre.shape[0]
    st.write(f"Number of Rows: {num_rows}")
    
    # Affichage des informations sur la base de donnée dans la sidebar
    st.sidebar.text("Data Types")
    st.sidebar.text(after_filtre.dtypes)
    
    # tri et récupération des colonnes selon leur type (Pourquoi ? j'en sais rien 
    # j'ai l'impression que ça sera utile plus tard ou donnera des idées)
    num, cat, bool = st.columns(3)
    
    # Récupération des colonnes de la base de donnée
    all_columns = after_filtre.columns
    
    # Initialiser une liste vide pour stocker les colonnes numériques
    numeric_columns = []

    
    for column in after_filtre.columns:
      # Verifier si la colonne contient des valeurs numériques
      if pd.api.types.is_numeric_dtype(after_filtre[column]) or pd.api.types.is_float_dtype(after_filtre[column]):
        numeric_columns.append(column)
    
    # Initialiser une liste vide pour stocker les colonnes catégoriques
    categorical_columns = []

   
    for column in after_filtre.columns:
      # Vérifier si la colonne contient des valeurs catégoriques
      if after_filtre[column].dtype == 'object':
        after_filtre[column] = after_filtre[column].astype('category')
        categorical_columns.append(column)
        
    
    # Initialiser une liste vide pour stocker les colonnes booléennes
    bool_columns = []

    for column in after_filtre.columns:
      if pd.api.types.is_bool_dtype(after_filtre[column]):
        bool_columns.append(column)
        
    # Description de la base de donnée
    st.subheader("Data Description")
    st.write(after_filtre.describe())
    
    # Visualisation des données
    st.subheader("Data Visualization")
    type_graphique= st.selectbox("Veuillez sélectionner le type de graphique que vous souhaitez afficher. ",("Correlation Matrix"  ,
                                                                                                             "Count Plot", "Pie Chart", 
                                                                                                             "Bar Plot", "Scatter plot", 
                                                                                                             "Line Chart", "Pair Plot", 
                                                                                                             "Histogram", "Box Plot"))
    
    if(type_graphique == "Correlation Matrix"):
      correlation_matrix(after_filtre)
    elif(type_graphique == "Count Plot"):
      count_plot(after_filtre)
    elif(type_graphique == "Pie Chart"):
      pie_chart(after_filtre)
    elif(type_graphique == "Bar Plot"):
      bar_plot(after_filtre)
    elif(type_graphique == "Scatter plot"):
      scatter_plot(after_filtre)
    elif(type_graphique == "Line Chart"):
      line_chart(after_filtre)
    elif(type_graphique == "Pair Plot"):
      pair_plot(after_filtre)
    elif(type_graphique == "Histogram"):
      histogram(after_filtre)
    elif(type_graphique == "Box Plot"):
      box_plot(after_filtre)
