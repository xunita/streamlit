import sys
import streamlit as st
import seaborn as sns
import altair as alt
# from awesome_table import AwesomeTable
# from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
# import pandasql as ps

import csv

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="MH Project",  # => Quick reference - Streamlit
    page_icon="📊",  # You can set a custom icon for the page
    layout="wide",   # Use "wide" layout for a wider page
    initial_sidebar_state="expanded",  # Set the initial state of the sidebar
)

# Custom CSS style
# custom_css = """
# <style>
#     .block-container {
#       display: flex !important;
#       justify-items: center !important;
#       justify-content: center !important;
#     }
# </style>
# """

# # Apply custom CSS
# st.markdown(custom_css, unsafe_allow_html=True)

def order_by(dataframe, column, ascending=True):
  # use a copy of the dataframe, to avoid modifying the original
  datacopy = dataframe.copy()
  return datacopy.sort_values(by=column, ascending=ascending)

def detect_separator(uploaded_file):
  # Convert bytes-like object to string
  content = uploaded_file.getvalue().decode('utf-8')

  # Use the first 1024 characters to detect the delimiter
  dialect = csv.Sniffer().sniff(content[:1024])

  return dialect.delimiter
st.session_state['refresh'] = 1
st.title("MH Project")
from io import StringIO
# separation_fichier=st.radio("Votre CSV est séparé par :",["Virgule", "Point Virgule", "tab"])

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    try:
      separator = detect_separator(uploaded_file)
      dataframe = pd.read_csv(uploaded_file, header=[0], sep=separator)
      
    except:
      st.error("Unable to detect separator. Please select a file with ','/';'/'\t' as separator.")
      sys.exit()

    # st.write(f"The detected separator in {uploaded_file} is: '{separator}'")

    #dataframe = pd.DataFrame(np.random.randn(10, 5),columns = ('col %d' % i for i in range(5)))
    # sidebar header
    st.sidebar.header("Search & metadata")
    col1 = st.sidebar.columns(1)
    col2= st.columns(1)
    copy = dataframe.copy()

    taille= len(copy.columns)
    filtre_par_colonne= st.sidebar.selectbox("Select a feature", copy.columns[0:taille])
    filtre_search_word= st.sidebar.text_input("Insert a " + "'"+filtre_par_colonne+"'" " value for search ", value=None, placeholder="Type a value...")
    filtre_ascending_des= st.sidebar.selectbox("Order by ",("Ascending"  ,"Descending"))
    filtre_null= st.sidebar.selectbox("Include null values",("Yes"  ,"No"))
    # filtre_par_ordre_colonne= st.sidebar.selectbox("Select a feature for order", dataframe.columns[0:taille])
    is_accessible = True

    if str(filtre_null)=='No' :
      is_accessible = False
      copy = copy.copy().dropna()
    else :
      is_accessible = True
      copy = copy

    rows_with_null = st.sidebar.checkbox('Show rows with only null values', value=False, key="accessible_checkbox_key", help="This checkbox can be toggled based on 'Include null values' state.", disabled=not is_accessible)

    filtre_null_stat= st.sidebar.selectbox("Null Values Statistics",copy.columns[0:taille])

    if filtre_search_word:
      # Check if the column is of type 'object' (likely a string)
        if copy[filtre_par_colonne].dtype == 'O':
            copy = copy[copy[filtre_par_colonne].astype(str).str.contains(filtre_search_word, case=False, na=False)]
        else:
            try:
                # Try to convert the search term to the data type of the column
                converted_search_term = type(copy[filtre_par_colonne].iloc[0])(filtre_search_word)
                copy = copy[copy[filtre_par_colonne] == converted_search_term]
            except (ValueError, TypeError):
                # Handle the case where conversion is not possible or fails
                pass

    if rows_with_null:
      copy = copy[copy.isnull().any(axis=1)]

    # Display number of null values and percentage for each variable
    null_stats = copy[str(filtre_null_stat)].isnull().sum()
    total_rows = len(copy)
    null_percentage = (null_stats / total_rows) * 100
    if total_rows == 0:
      st.sidebar.write(f"{str(filtre_null_stat)}: {null_stats}/{total_rows} row(s), about 0.00%")
    else:
      st.sidebar.write(f"{str(filtre_null_stat)}: {null_stats}/{total_rows} row(s), about {null_percentage:.2f}%")



    if str(filtre_ascending_des)=='Ascending' :
      copy = order_by(copy, filtre_par_colonne, ascending=True)
      st.session_state['refresh'] += 1
    else :
      copy = order_by(copy, filtre_par_colonne, ascending=False)
      st.session_state['refresh'] += 1



      # "SELECT * FROM dataframe  where  "str(filtre_par_colonne)" = "int(filtre_search_word)" Order by " +str(filtre_par_ordre_colonne)+ " DESC "
    after_filtre=copy
    #affichage de la base de donnée
    st.dataframe(after_filtre, use_container_width=True, hide_index=True)
    num_rows = after_filtre.shape[0]
    st.write(f"Number of Rows: {num_rows}")
    #les types des données
    st.sidebar.text("Data Types")
    st.sidebar.text(after_filtre.dtypes)

    #AwesomeTable(dataframe,show_order=True, show_search=True, show_search_order_in_sidebar=True)
    #choix=AwesomeTable.show_search()
    
    num, cat, bool = st.columns(3)
    
    # Initialize an empty list to store numeric/float columns
    all_columns = []
    numeric_columns = []

    # Iterate through DataFrame columns
    for column in after_filtre.columns:
      all_columns.append(column)
      # Check if the column contains numeric or float values
      if pd.api.types.is_numeric_dtype(after_filtre[column]) or pd.api.types.is_float_dtype(after_filtre[column]):
        # Append the column to the list
        numeric_columns.append(column)

      # Print the list of numeric/float columns
    # num.write(numeric_columns)
    
    # Initialize an empty list to store categorical columns
    categorical_columns = []

    # Iterate through DataFrame columns
    for column in after_filtre.columns:
      # Check if the column contains categorical values
      if after_filtre[column].dtype == 'object':
        # Convert object type columns to categorical
        after_filtre[column] = after_filtre[column].astype('category')
        # Append the column to the list of categorical columns
        categorical_columns.append(column)
        
    # cat.write(categorical_columns)
    
    # Initialize an empty list to store boolean columns
    bool_columns = []

    # Iterate through DataFrame columns
    for column in after_filtre.columns:
      # Check if the column contains boolean values
      if pd.api.types.is_bool_dtype(after_filtre[column]):
        bool_columns.append(column)
        
    # bool.write(bool_columns)
    st.subheader("Data Description")
    st.write(after_filtre.describe())
    st.subheader("Data Visualization")
    var4, var2, var3 = st.columns(3)
    var1, var5, var6 = st.columns(3)
    var7, var8, var9 = st.columns(3)
    
    v7 = after_filtre.copy()
    
    var7.subheader("Bar Plot")
    fig_bb, ax_bb = plt.subplots(figsize=(10, 8))
    # Select x and y columns for customization
    x_columnbc = var7.selectbox("Select X-axis column Bar Plot:", v7.columns)
    y_columnbc = var7.selectbox("Select Y-axis column Bar Plot:", v7.columns)

    # Create a custom DataFrame for the line chart
    sns.barplot(x=x_columnbc, y=y_columnbc, data=v7, ax=ax_bb)

    # Plot the line chart
    var7.pyplot(fig_bb)
    
    
    v9 = after_filtre.copy()
    var9.subheader("Count Plot")
    fig_cp, ax_cp = plt.subplots(figsize=(10, 8))
    # Select x and y columns for customization
    x_columncp = var9.selectbox("Select column Count Plot:", v9.columns)
    # y_columnbc = var7.selectbox("Select Y-axis column Bar Plot:", after_filtre.columns)

    # Create a custom DataFrame for the line chart
    sns.countplot(data=v9[x_columncp], ax=ax_cp)
    # sns.barplot(x=x_columnbc, y=y_columnbc, data=after_filtre, ax=ax_bb)

    # Plot the line chart
    var9.pyplot(fig_cp)
    
    
    
    v8 = after_filtre.copy()
    var8.subheader("Pie Chart")
    fig_p, ax_p = plt.subplots(figsize=(10, 8))
    # Select x and y columns for customization
    x_columnp = var8.selectbox("Select column Pie Chart:", v8.columns)

    # Create a custom DataFrame for the line chart
    var8.area_chart(v8[x_columnp].value_counts())
    
    
    v4 = after_filtre.copy()
    fig_cor, ax_cor = plt.subplots(figsize=(10, 8))
    
    var4.subheader("Correlation Matrix")
    corr_matrix = v4[numeric_columns].corr()
    sns.heatmap(corr_matrix[numeric_columns], annot=True, cmap='coolwarm', ax=ax_cor)
    var4.pyplot(fig_cor)
    
    v5 = after_filtre.copy()
    var5.subheader("Scatter plot")
    fig_sc, ax_sc = plt.subplots(figsize=(8, 8))
    # Select X and Y axes
    x_axis = var5.selectbox("Select X-axis for Scatter:", v5.columns)
    y_axis = var5.selectbox("Select Y-axis for Scatter:", v5.columns)
    sns.scatterplot(x=v5[x_axis], y=v5[y_axis], ax=ax_sc)
    ax_sc.set_title("Scatter Plot")
    # Set X and Y axis labels
    ax_sc.set_xlabel(x_axis)
    ax_sc.set_ylabel(y_axis)

    # Display the plot in Streamlit
    var5.pyplot(fig_sc)
    
    v6 = after_filtre.copy()
    var6.subheader("Line Chart")

    # Select x and y columns for customization
    x_columnl = var6.selectbox("Select X-axis column Line Chart:", numeric_columns)
    y_columnl = var6.selectbox("Select Y-axis column Line Chart:", numeric_columns)

    # Create a custom DataFrame for the line chart
    line_chart_data = v6[[x_columnl, y_columnl]]

    # Plot the line chart
    var6.line_chart(line_chart_data)
    
    v1 = after_filtre.copy()
    # Select x and y columns for customization
    var1.subheader(f"Pair Plot")
    x_column = var1.selectbox("Select X-axis column:", v1.columns)
    y_column = var1.selectbox("Select Y-axis column:", v1.columns)
    
    # Check data types for x and y columns
    # x_is_categorical = after_filtre[x_column].dtype == 'category'
    # y_is_categorical = after_filtre[y_column].dtype == 'category'

    # Customize pair plot based on data types
    # if x_is_categorical and y_is_categorical:
    #     st.subheader(f"Pair Plot (Customized X: {x_column}, Y: {y_column}, Vertical Orientation)")
    #     sns.pairplot(after_filtre, x_vars=[x_column], y_vars=[y_column], palette='husl', height=7)
    # else:
    #   st.subheader(f"Pair Plot (Customized X: {x_column}, Y: {y_column})")
    #   sns.pairplot(after_filtre, x_vars=[x_column], y_vars=[y_column], height=7)

    # Display pair plot with customized x and y columns
    var1.write(f"Customized X: {x_column}, Y: {y_column}")
    # if after_filtre[column_name].dtype == 'category':  # Check if the column is categorical
    #   ax.hist(after_filtre[column_name], orientation='horizontal', bins='auto')
    # else:
    #   ax.hist(after_filtre[column_name])
    sns.pairplot(v1, x_vars=[x_column], y_vars=[y_column], height=7)
    var1.pyplot()
    
    
    v2 = after_filtre.copy()
    var2.subheader(f"Histogram")
    column_name = var2.selectbox("Select a column:", v2.columns)
    fig_hist, ax_hist = plt.subplots(figsize=(6, 6))
    # Plot histogram with vertical orientation for categorical variables
    if v2[column_name].dtype == 'category':  # Check if the column is categorical
      ax_hist.hist(v2[column_name], orientation='horizontal', bins='auto')
    else:
      ax_hist.hist(v2[column_name])
    var2.pyplot(fig_hist)
    
    
    v3 = after_filtre.copy()
    var3.subheader("Box Plot")
    # Select x and y columns for customization
    x_columnb = var3.selectbox("Select X-axis column for Box Plot:", v3.columns)
    y_columnb = var3.selectbox("Select Y-axis column for Box Plot:", v3.columns)

    # Display box plot with customized x and y columns
    var3.write(f"Customized X: {x_columnb}, Y: {y_columnb}")
    fig_box, ax_box = plt.subplots(figsize=(8, 8))
    sns.boxplot(x=x_columnb, y=y_columnb, data=v3, ax=ax_box)
    # box_name = var3.selectbox("Select a column for Box Plot:", after_filtre.columns)
    # sns.boxplot(x=after_filtre[box_name])
    var3.pyplot(fig_box)
    # hue_column = st.selectbox("Select a column for customization:", after_filtre.columns)

    # # Display pair plot with customized hue
    # st.subheader(f"Pair Plot (Customized by {hue_column})")
    # sns.pairplot(after_filtre, hue=hue_column)
    # st.pyplot()
    
    # st.title("Pair Plot")
    # sns.pairplot(after_filtre)
    # st.pyplot()
    
    # st.title("Data Visualization")
    # st.subheader("Bar Chart")
    # # st.write(after_filtre.columns)
    # st.bar_chart(after_filtre, height=500, x="Acceleration", y=categorical_columns)
    # # st.bar_chart(after_filtre['Category'].value_counts())
    
    # option1 = st.checkbox('View with Line chart')
    # if option1:
    #   st.line_chart(after_filtre)
    # option2 = st.checkbox('View with Area chart')
    # if option2:
    #   st.area_chart(after_filtre)
    # option3 = st.checkbox('View with Bar chart')
    # if option3:
    #   st.bar_chart(after_filtre)
    # option4 = st.checkbox('View with Mark Point')
    # if option4:
    #   chart= alt.Chart(after_filtre).mark_tick().encode(x=str(filtre_par_colonne)).interactive()
    #   st.altair_chart(chart, theme="streamlit", use_container_width= False)
    #   chart2= alt.Chart(after_filtre).mark_bar().encode(x=alt.X(filtre_par_colonne, bin=True), y=filtre_par_colonne, color=filtre_par_colonne)
    #   st.altair_chart(chart2, theme="streamlit", use_container_width= False)

