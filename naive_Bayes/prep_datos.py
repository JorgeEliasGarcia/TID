#!/usr/bin/env python3

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC

# Variable global para los datos
mydata = pd.read_csv("../homeLoanAproval.csv")

# Función para los Cambios de tipo.
def cambios_de_tipo(): 
  mydata['Married'] = mydata['Married'] == "Yes"  # Married será de tipo boolean. Casado o No.
  mydata['SelfEmployed'] = mydata['SelfEmployed'] == "Yes" # Cambiamos SelfEmployed también a tipo boolean.


# Necesitamos identificar que variables podemos emplear para la agrupación en clusters. Estas serán aquellas que tienen valor numérico y ningún valor a nulo.
def identificar_variables_validas_para_agrupacion():
  global mydata
  null_counts = mydata.isnull().sum() # Calculamos cuántos valores nulos hay en cada variable 
  
  # Para poder agrupar por más variables, si una variable solo toma valores nulos 8 veces o menos, eliminamos esas filas.
  for column in mydata.columns:
    if column != 'Loan_ID' and null_counts[column] <= 8: # Si el valor nulo es el ID no importa. 
      mydata = mydata.dropna(subset=[column])
  
  # Seleccionamos columnas sin valores nulos y numéricas, excluyendo 'Loan_ID'
  selected_columns = []
  for column in mydata.columns:
    if column != 'Loan_ID' and null_counts[column] == 0: # Verificamos si la columna no tiene valores nulos y no es 'Loan_ID'
      if mydata[column].dtype in ['int64', 'float64']:  # Verificamos si la columna es numérica
        selected_columns.append(column)

  return selected_columns



# Hacemos la agrupación en clusters, haciendo uso del algoritmo kMeans. 
def hacer_agrupacion_clusters(selected_columns):
  grouped_data = mydata[selected_columns]   # Agrupar por las columnas seleccionadas
  # Aplicar K-means
  kmeans = KMeans(n_clusters=5)  # Usaremos 5 clusters
  kmeans.fit(grouped_data)
  # Asignamos clusters a cada fila en el DataFrame original
  mydata['Cluster'] = kmeans.labels_
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
  approved_loans = len(mydata[mydata["LoanStatus"] == 'Y'])
  not_approved_loans = len(mydata[mydata["LoanStatus"] == 'N'])
  return (abs(approved_loans - not_approved_loans))


# Función para balancear las clases. Para ello, añadiremos más filas con la clase menos representada, usando el algoritmo SMOTE. 
def balancear_clases(): 
  global mydata 
  if (diferencia_balance_clases() > len(mydata) * 0.30): # Comprobamos si están desbalanceadas. Si la diferencia es mayor al 30% del total de filas
    print("Balanceando clases...")

def main(): 
  global mydata
  cambios_de_tipo()
  tratamiento_valores_nulos()
  print(len(mydata))
  balancear_clases()
  print(len(mydata))
  

main(); # Llamamos a la función principal
