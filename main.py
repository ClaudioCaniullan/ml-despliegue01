# importar utilitarios en la carpeta raiz
from utils import Utils
from models import Models

import joblib
import numpy as np

if __name__ == "__main__":
    
    # importar clases de los utilitarios
    utils = Utils()
    models = Models()
    
    # cargar data desde carpeta in
    data = utils.load_from_csv('./in/data2.csv')

    # definir features y target
    X, y = utils.features_target(data, ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name', 'Transported_False','Transported_True'],['Transported_True'])

    # entrenar modelo
    models.grid_training(X,y)

    # realizar prediction
    model = joblib.load('./models/best_model.pkl')
    # construir X_test de prueba para prediccion
    X_test = np.array([39.0,0.0,0.0,0.0,0.0,0.0])
    prediction = model.predict(X_test.reshape(1,-1))
    print("prediccion score con SVR:",prediction)
    



    # probar salida en consola con
    # print(data)

    
