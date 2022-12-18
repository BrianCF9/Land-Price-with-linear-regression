import requests
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('Casa.html')


@app.route("/Mapa")
def mapa():
    return render_template('Mapa.html')


@app.route("/TasadorOnline")
def Tasador():
    return render_template('TasadorOnline.html')


@app.route("/prediccion", methods=["POST"])
def prediccion():

    # cargamos los datos de entrada
    data = pd.read_csv("fechas.csv", delimiter=";")
    # veamos cuantas dimensiones y registros contiene

    #data.drop('SupTotal', inplace=True, axis=1)
    #data.drop('SupUtil', inplace=True, axis=1)
    # data.head()

    # Vamos a RECORTAR los datos en la zona donde se concentran más los puntos
    # esto es en el eje X: entre 0 y 3.500
    # y en el eje Y: entre 0 y 80.000

    filtered_data = data[(data['SupUtil'] > 10) & (data['SupTotal'] > 10)]
    #filtered_data = data[ (data['Comuna'] =="Viña Del Mar")  ]
    # filtered_data=data

    # Visualizamos rápidamente las caraterísticas de entrada

    dataX2 = pd.DataFrame()
    dataX2["Habitaciones"] = filtered_data["Habitaciones"]
    dataX2["Baños"] = filtered_data["Baños"]
    #dataX2["Lat"] = filtered_data["Lat"]
    #dataX2["Long"] = filtered_data["Long"]
    dataX2["SupTotal"] = filtered_data["SupTotal"]
    dataX2["SupUtil"] = filtered_data["SupUtil"]
    #dataX2["Hospital"] = filtered_data["Hospital"]
    #dataX2["Comisaria"] = filtered_data["Comisaria"]
    #dataX2["Banco"] = filtered_data["Banco"]
    #dataX2["Farmacia"] = filtered_data["Farmacia"]
    XY_train = np.array(dataX2)
    z_train = filtered_data['Precio(CLP)'].values

    # Creamos un nuevo objeto de Regresión Lineal
    regr2 = linear_model.LinearRegression()

    # Entrenamos el modelo, esta vez, con 2 dimensiones
    # obtendremos 2 coeficientes, para graficar un plano
    regr2.fit(XY_train, z_train)

    # Hacemos la predicción con la que tendremos puntos sobre el plano hallado
    z_pred = regr2.predict(XY_train)

    # Los coeficientes
    print('Coefficients: \n', regr2.coef_)
    # Error cuadrático medio
    print("Mean squared error: %.2f" % mean_squared_error(z_train, z_pred))
    # Evaluamos el puntaje de varianza (siendo 1.0 el mejor posible)
    print('Variance score: %.2f' % r2_score(z_train, z_pred))

    # Si quiero predecir cuántos "Shares" voy a obtener por un artículo con:
    # 2000 palabras y con enlaces: 10, comentarios: 4, imagenes: 6
    # según nuestro modelo, hacemos:
    data = request.json

    z_Dosmil = regr2.predict([[int(data["Habitaciones"]), int(
        data["Baños"]), int(data["SupTotal"]), int(data["SupUtil"])]])
    #z_Dosmil = regr2.predict([[3,2,1,0,0,1]])

    return jsonify(int(z_Dosmil))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True, threaded=True)
