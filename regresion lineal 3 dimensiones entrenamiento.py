# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:23:55 2022

@author: alber
"""

#Librerías y módulos necesarios
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#Cargamos los datos y creamos el dataframe
casas_boston=load_boston()
casas = pd.DataFrame(casas_boston.data, columns=casas_boston.feature_names)
casas['MEDV'] = casas_boston.target 


#Preparación de los dataframe para la regresión
X = pd.DataFrame(np.c_[casas['LSTAT'], casas['RM']], columns = ['LSTAT','RM'])
y=casas['MEDV']


#Realizamos el entrenamiento y lanzamos las predicciones
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=20)
lr=LinearRegression().fit(X_train,y_train)
y_pred=lr.predict(X_test)


#Dibujamos el hiperplano de regresión. Representamos en primer lugar los puntos
fig = plt.figure(figsize=(18,15)) 
ax = fig.add_subplot(111, projection='3d')
ax.scatter(casas['LSTAT'],casas['RM'],casas['MEDV'],c='b') 
ax.set_xlabel("LSTAT",fontsize=15) 
ax.set_ylabel("RM",fontsize=15) 
ax.set_zlabel("MEDV",fontsize=15)


#Ajustamos el espacio de representación y representamos el hiperplano en sí
lstat_sup = np.arange(0, 40, 1)   
rm_sup = np.arange(0, 10, 1)   
lstat_sup, rm_sup = np.meshgrid(lstat_sup, rm_sup)
z_medv = lambda x,y: (lr.intercept_ + lr.coef_[0] * x + lr.coef_[1] * y)
ax.plot_surface(lstat_sup, rm_sup, z_medv(lstat_sup,rm_sup),rstride=1, cstride=1, 
                color='None',alpha = 0.4)


#Valores de los pesos del modelo de regresión
print("Coeficiente w0:",lr.intercept_)
print("Coeficiente w1:",lr.coef_[0])
print("Coeficiente w2:",lr.coef_[1])


#Valor de los coeficientes de determinación y del error cuadrático medio
print('Coeficiente de determinación del conjunto de entrenamiento: %.4f' % lr.score(X_train,
                                                                                 y_train))
print('Coeficiente de determinación del conjunto de pruebas: %.4f' % lr.score(X_test,
                                                                                 y_test))
print("Error cuadrático medio:",round(mean_squared_error(y_test, y_pred),4))

"""
En este código se pretende entrenar un modelo de regresión lineal multiple (3 dimensiones)
para predecir el precio de la vivienda en boston  en base a diferentes variables obtenidas 
de un dataset real (boston_housing). A su vez, este código implementa unos indicadores a modo
de coeficientes para, una vez entrenado el modelo, evaluar su exactitud. 

El área al que pertenecería este caso sería al de aprendizaje automático o "machine learning".

Python es un lenguaje que destaca por su sencillez a la hora de tratar gran cantidad de datos. 
Uno de los papeles fundamentales de python en este caso práctico, en el que enfrentamos precdir un
resulado en base a un "dataset", es la facilidad de poder transformar, seccionar, reorganizar y
visualizar todo este conjunto masivo de datos. Para ello, no solo se emplean comandos propios del
lenguaje, sino que se cuenta con librerías ampliamente documentadas como puede ser "matplotlib.pyplot",
empleada para la construcción y visualización de gráficas; numpy, para el desarrollo de cálculos 
matemáticos; pandas, librería que permite procesar datos, transformarlos, crear los dataframes
con los que se trabajarán, además de poder trabajar con archivos CSV; o scikitlearn, para la
implementación de algoritmos de machine learning ya optimizados.

Python en sí nos permite poner en común todas estas herramientas en forma de librerías y por lo 
tanto generar un código que englobe, a fin de cuentas, una solución (lo más automatizada posible)
para resolver uno o varios problemas.

El fin de este programa no sería la automatización pero si la resolución de un problema específico,
dado para unos datos específicos, con el menor coste de tiempo y recursos posible.
"""