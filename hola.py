import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

df.shape #Tamaño de la tabla (df.shape[0] es el número de filas y df.shape[1] es el número de columnas)
print("Cantidad de Muestras y Columnas")
print(df.shape)
print(df.head(5))
#print(df.dtypes) #df.dtypes #Muestra el tipo de dato que se almacena en cada columna de la tabla

# Modificamos la variable 'TotalCharges' a tipo numérico
print("Modificando campo Total Charges a numerico")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)


df.columns=df.columns.str.lower().str.replace(' ','_')
#print()
#print(df.dtypes)
string_columns = list(df.dtypes[df.dtypes =='object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.drop('customerid',axis=1,inplace=True)#elminamos la columna customerID
#print()
#print(df.dtypes)


#Separar los datos en entrenamiento, validación y test
print("Seprando los datos de training validacion y test")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train, test_size=0.33, random_state=11)

#Eliminar la columna churn de la tabla de características
print("Eliminando columna churn de cada dataset")
y_train = df_train["churn"].values
df_train.drop(["churn"],inplace=True,axis = 1)

y_val = df_val["churn"].values
df_val.drop(["churn"],inplace=True,axis = 1)

y_test = df_test["churn"].values
df_test.drop(["churn"],inplace=True,axis = 1)

print(f"{df_train.shape[0]} Muestras de entrenamiento")
print(f"{df_val.shape[0]} Muestras de validación")
print(f"{df_test.shape[0]} Muestras de prueba")

#Índices de las columnas numéricas y categ+oricas
cat_cols = df_train.select_dtypes(include=object).columns
num_cols = df_train.select_dtypes(include=np.number).columns



