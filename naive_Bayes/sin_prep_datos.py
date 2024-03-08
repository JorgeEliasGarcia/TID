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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Variable global para los datos
mydata = pd.read_csv("../homeLoanAproval.csv")


# Creamos columnas binarias, para las variables catgóricas. Usamos OneHotEnconder 
pipeline = ColumnTransformer(
    [   
        ("text", OneHotEncoder(), ["Gender", "Education", "PropertyArea", "Married", "SelfEmployed", "Dependents", "CoapplicantIncome"]),
        ("numeric", StandardScaler(), ["ApplicantIncome", "LoanAmount", "LoanAmountTerm" ])
    ], 
)

# Función para la división de los datos
def division_datos_entrenamiento_prueba(): 
  no_loan_status = mydata.drop("LoanStatus", axis=1)
  preprocessed_dataset = pipeline.fit_transform(no_loan_status)
  X_train, X_test, y_train, y_test = train_test_split ( preprocessed_dataset, mydata["LoanStatus"], test_size=0.2, random_state=42 )
  return X_train, X_test, y_train, y_test


# Función para implementar el clasificador k-NN
def clasificador_KNN():
  X_train, X_test, y_train, y_test = division_datos_entrenamiento_prueba()  # Dividimos los datos en conjuntos de entrenamiento y prueba. IMPORTANTE, estamos manteniendo la proporción de clases. 
  knn_classifier = KNeighborsClassifier(n_neighbors=10)   # Instanciar el clasificador KNN
  knn_classifier.fit(X_train, y_train)   # Instanciamos y entrenamos el clasificador KNN
  y_pred = knn_classifier.predict(X_test)   # Predecir las etiquetas en el conjunto de prueba
  accuracy = accuracy_score(y_test, y_pred)  # Evaluamos el rendimiento del clasificador
  print("Claisificador KNN: \nAccuracy:", accuracy)


# FUnción para implementar el clasificador de árbol de clasificación
def arbol_clasificacion(): 
  X_train, X_test, y_train, y_test = division_datos_entrenamiento_prueba()  # Dividimos los datos en conjuntos de entrenamiento y prueba. IMPORTANTE, estamos manteniendo la proporción de clases.
  tree_classifier = DecisionTreeClassifier(random_state=42) # Instanciamos el clasificador de árbol de decisión
  # scores = cross_val_score(tree_classifier, X_train, y_train, cv=5)  # Utilizamos validación cruzada con 5 folds
  tree_classifier.fit(X_train, y_train) # Entrenamos
  y_pred = tree_classifier.predict(X_test) # Predecimos
  accuracy = accuracy_score(y_test, y_pred) # Evaluamos el rendimiento
  print("Arbol Clasificación: \nAccuracy:", accuracy) 


# Función para implementar el clasificador naive Bayes. Utilizamos la versión que permite utilizar variables categóricas.
def clasificador_naive_bayes(): 
  X_train, X_test, y_train, y_test = division_datos_entrenamiento_prueba() 
  naive_bayes_classifier = GaussianNB() # Instanciamos el clasificador naive Bayes
  naive_bayes_classifier.fit(X_train.toarray(), y_train) # Convertimos X_train a formato denso
  y_pred = naive_bayes_classifier.predict(X_test.toarray()) # Convertimos X_test a formato denso
  accuracy = accuracy_score(y_test, y_pred)
  print("Naive Bayes: \nAccuracy: ", accuracy)
  

def main(): 
  global mydata
  # Eliminamos todos los nulos para usar los clasificadores 
  mydata = mydata.dropna() 
  clasificador_KNN()
  arbol_clasificacion()
  clasificador_naive_bayes()


main(); # Llamamos a la función principal
