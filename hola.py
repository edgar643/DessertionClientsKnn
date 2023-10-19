import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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

#Eliminar la columna churn de la tabla de características (Targe )
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

print()
print("Obteniendo columnas categorias y numericas por separado")
#Índices de las columnas numéricas y categ+oricas
cat_cols = df_train.select_dtypes(include=object).columns
num_cols = df_train.select_dtypes(include=np.number).columns


numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
) 


# Crear un clasificador KNN con diferentes números de vecinos
n_neighbors = [1,3,5,7,9,11,13,15]
accuracy = []


for k in n_neighbors:
  clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors=k))])

  # Entrenar el clasificador con los datos de entrenamiento
  
  clf.fit(df_train, y_train)

  # Evaluar el rendimiento del clasificador en los datos de validación
  accuracy.append(clf.score(df_val, y_val))

#Re-entrenar el modelo con los datos de entrenamiento y validación para el valor de k óptimo
accuracyMax = np.max(accuracy)
k_opt = n_neighbors[np.argmax(accuracy)]
print(f"El número óptimo de vecinos es {k_opt} y precision maxima es {accuracyMax:0.2f}")
X_train = pd.concat([df_train,df_val],axis=0)
y_train = np.hstack([y_train,y_val])

# Entrenar el clasificador con los datos de entrenamiento
clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors=k_opt))])
clf.fit(X_train, y_train)
#Validar el rendimiento del clasificador con los datos de prueba
accuracy = clf.score(df_test, y_test)

print(f"La tasa de acierto de clasificación en el conjunto de test es {accuracy:0.2f}")

print("Registro a probar")
print(df_test.iloc[[999],:])
print("Prediccion")
print(clf.predict(df_test.iloc[[999],:]))
print("Valor real")
print(y_test[999])
