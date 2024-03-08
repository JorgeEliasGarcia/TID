#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Variable global para los datos
mydata = pd.read_csv("../homeLoanAproval.csv")

# Función para los Cambios de tipo.
def cambios_de_tipo(): 
  mydata['CoapplicantIncome'] = mydata['CoapplicantIncome'].str.replace('.', '')  # Eliminamos los puntos de la columna 'CoapplicantIncome', ya que si no, no podremos pasarla a tipo entero.
  mydata['CoapplicantIncome'] = mydata['CoapplicantIncome'].astype(int)
  mydata['Dependents'] = mydata['Dependents'].replace('3+', 3) # Asignamos un valor numérico a '3+'. Le asigno un 3, porque lo más probable es que si tienen 3 hijos o más tengan 3 hijos.
  mydata['Dependents'] = pd.to_numeric(mydata['Dependents']) # Convertir la columna a tipo numérico
  mydata['Married'] = mydata['Married'] == "Yes"  # Married será de tipo boolean. Casado o No.
  mydata['SelfEmployed'] = mydata['SelfEmployed'] == "Yes" # Cambiamos SelfEmployed también a tipo boolean.
 # mydata['LoanStatus'] = mydata['LoanStatus'] == "Y"  # LoanStatus también será de tipo boolean. Aprobado o No. 
  


# Implementamos la detección y eliminación de outliers. 
def detectar_y_eliminar_outliers(data, contamination = 0.05): # Esperamos que aproximadamente el 5% de los datos sean outliers. 
  data_numeric = data.select_dtypes(include=[float, int])  # Seleccionamos solo columnas numéricas, para el análisis de outliers. 
  data_no_nulls = data_numeric.dropna()   # Guardamos una copia de los datos con valores no nulos. No eliminamos filas con nulos, más tarde les imputaremos un valor.
  modelo_iforest = IsolationForest(contamination=contamination)  
  modelo_iforest.fit(data_no_nulls)   # Entrenamos el modelo
  predicciones = modelo_iforest.predict(data_no_nulls)  # Predicción de outliers
  outliers_indices = data_no_nulls[predicciones == -1].index  # Índices de los outliers detectados
  if len(outliers_indices) < len(data) * 0.05:   # Eliminamos los outliers detectados si representan menos del 5% de la base de datos
    data_cleaned = data.drop(outliers_indices)
  else:
    data_cleaned = data

  return data_cleaned



# Necesitamos identificar que variables podemos emplear para la agrupación en clusters. Estas serán aquellas que tienen valor numérico y ningún valor a nulo.
def identificar_variables_validas_para_agrupacion():
  global mydata
  null_counts = mydata.isnull().sum() # Calculamos cuántos valores nulos hay en cada variable 
  
  # Para poder agrupar por más variables, si una variable solo toma valores nulos 13 veces o menos, eliminamos esas filas.
  for column in mydata.columns:
    if null_counts[column] <= 15: # Si el valor nulo es el ID no importa. 
      mydata = mydata.dropna(subset=[column])
  
  # Seleccionamos columnas sin valores nulos y numéricas, excluyendo 'Loan_ID'
  selected_columns = []
  for column in mydata.columns:
    if null_counts[column] == 0: # Verificamos si la columna no tiene valores nulos y no es 'Loan_ID'
      if mydata[column].dtype in ['int64', 'float64']:  # Verificamos si la columna es numérica
        selected_columns.append(column)

  return selected_columns



# Hacemos la agrupación en clusters, haciendo uso del algoritmo kMeans. 
def hacer_agrupacion_clusters(selected_columns):
  grouped_data = mydata[selected_columns]   # Agrupar por las columnas seleccionadas
  kmeans = KMeans(n_clusters=10)  #  Aplicamos kMeans, usando 15 clusters
  kmeans.fit(grouped_data)
  mydata['Cluster'] = kmeans.labels_   # Asignamos clusters a cada fila en el DataFrame original
  return grouped_data


# Imputar los valores nulos basados en el cluster al que pertenecen
def imputar_valores_nulos(grouped_data):
  # Imputar los valores nulos basados en el cluster al que pertenecen
  imputer = SimpleImputer(strategy='mean', missing_values=pd.NA)
  imputer.fit(grouped_data)
  
  # Iterar sobre las columnas
  for column in mydata.columns:
    if mydata[column].isnull().sum() > 0: # Si la columna tiene valores nulos
      if mydata[column].dtype == 'object':
        # Si es una variable categórica
        for index, row in mydata.iterrows():
          if pd.isnull(row[column]):
            cluster = row['Cluster'] # Obtener el clúster de la fila actual
            cluster_mode = mydata[(mydata['Cluster'] == cluster)][column].mode()[0]  # Filtrar por el cluster al que pertenece y calculamos la moda
            mydata.at[index, column] = cluster_mode
      else:
        for index, row in mydata.iterrows():  # Si es una variable numérica
          if pd.isnull(row[column]):
            cluster = row['Cluster']  # Obtener el clúster de la fila actual
            cluster_mean = mydata[(mydata['Cluster'] == cluster)][column].mean() #Filtramos por el cluster al que pertenece y calculamos la media
            mydata.at[index, column] = cluster_mean


# Función donde tratamos los valores nulos. 
def tratamiento_valores_nulos():
  selected_columns = identificar_variables_validas_para_agrupacion()
  grouped_data = hacer_agrupacion_clusters(selected_columns)
  imputar_valores_nulos(grouped_data)


# Comprobamos si las clases están correctamente balanceadas
def diferencia_balance_clases():
  approved_loans = len(mydata[mydata["LoanStatus"] == "Yes"])
  not_approved_loans = len(mydata[mydata["LoanStatus"] == "No"])
  return (abs(approved_loans - not_approved_loans))


# Función para balancear las clases. Para ello, añadiremos más filas con la clase menos representada, usando el algoritmo SMOTE. 
def balancear_clases(): 
  global mydata 
  if (diferencia_balance_clases() > len(mydata) * 0.3): # Comprobamos si están desbalanceadas. Si la diferencia es mayor al 30% del total de filas
    X = mydata.drop("LoanStatus", axis=1)  # Características
    y = mydata["LoanStatus"]  # Etiquetas
    categorical_indices = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']  # Obtenemos los índices de columnas categóricas
    smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)  # Creamos una instancia de SMOTENC, para aplicar el algoritmo SMOTE, pero teniendo en cuenta las variables categóricas.
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)  # Aplicamos SMOTENC para balancear las clases
    mydata = X_resampled.join(y_resampled.rename("LoanStatus")) # Actualizamos el conjunto de datos con las nuevas muestras balanceadas
    

# Función para encapsular todo el tratamiento de los datos. Es decir, la preparación de los datos
def prepracion_datos(): 
  global mydata
  cambios_de_tipo()
  mydata.drop('Loan_ID', axis=1, inplace=True)   # Eliminamos la columna LOAD_ID, ya que no aporta información relevante para el análisis.
  mydata = detectar_y_eliminar_outliers(mydata) # Detectamos los outliers antes de agrupar, ya que si no obtendríamos clusters solo para los outliers y resultaría en malas agrupaciones. 
  tratamiento_valores_nulos()
  mydata.drop('Cluster', axis=1, inplace=True) # Eliminamos la columna Cluster, ya que no la necesitamos para el análisis posterior. 
  balancear_clases() 



num_attributes = mydata.select_dtypes(include=["int64", "float64"]).columns.to_numpy()

# Creamos columnas binarias, para las variables catgóricas. Usamos OneHotEnconder 
pipeline = ColumnTransformer(
    [   
        ("text", OneHotEncoder(), ["Gender", "Education", "PropertyArea"]),
        ("numeric", StandardScaler(), num_attributes)
    ]
)

# Función para la división de los datos
def division_datos_entrenamiento_prueba(data): 
  preprocessed_dataset = pipeline.fit_transform(data)
  X_train, X_test, y_train, y_test = train_test_split ( preprocessed_dataset, mydata["LoanStatus"], test_size=0.2, random_state=42 )
  return X_train, X_test, y_train, y_test


# Función para implementar el clasificador k-NN
def clasificador_KNN():
  X_train, X_test, y_train, y_test = division_datos_entrenamiento_prueba(mydata)  # Dividimos los datos en conjuntos de entrenamiento y prueba. IMPORTANTE, estamos manteniendo la proporción de clases. 
  knn_classifier = KNeighborsClassifier(n_neighbors=5)   # Instanciar el clasificador KNN
  knn_classifier.fit(X_train, y_train)   # Instanciamos y entrenamos el clasificador KNN
  y_pred = knn_classifier.predict(X_test)   # Predecir las etiquetas en el conjunto de prueba
  accuracy = accuracy_score(y_test, y_pred)  # Evaluamos el rendimiento del clasificador
  print("Claisificador KNN: \nAccuracy:", accuracy)


# FUnción para implementar el clasificador de árbol de clasificación
def arbol_clasificacion(): 
  X_train, X_test, y_train, y_test = division_datos_entrenamiento_prueba(mydata)  # Dividimos los datos en conjuntos de entrenamiento y prueba. IMPORTANTE, estamos manteniendo la proporción de clases.
  tree_classifier = DecisionTreeClassifier(random_state=42) # Instanciamos el clasificador de árbol de decisión
  tree_classifier.fit(X_train, y_train) # Entrenamos
  y_pred = tree_classifier.predict(X_test) # Predecimos
  accuracy = accuracy_score(y_test, y_pred) # Evaluamos el rendimiento
  print("Arbol Clasificación: \nAccuracy:", accuracy) 


# Función para implementar el clasificador naive Bayes. Utilizamos la versión que permite utilizar variables categóricas.
def clasificador_naive_bayes(): 
  X_train, X_test, y_train, y_test = division_datos_entrenamiento_prueba(mydata) 
  naive_bayes_classifier = GaussianNB() # Instanciamos el clasificador naive Bayes
  naive_bayes_classifier.fit(X_train, y_train)
  y_pred = naive_bayes_classifier.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Naive Bayes: \nAccuracy: ", accuracy)
  

def main(): 
  global mydata
  mydata.dropna()
  clasificador_KNN()
  arbol_clasificacion()
  clasificador_naive_bayes()


main(); # Llamamos a la función principal
