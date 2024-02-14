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
    data = utils.load_from_csv('./in/felicidad.csv')

    # definir features y target
    X, y = utils.features_target(data, ['country','score','rank'],['score'])

    # entrenar modelo
    models.grid_training(X,y)

    # realizar prediction
    model = joblib.load('./models/best_model.pkl')
    # construir X_test de prueba para prediccion
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_test.reshape(1,-1))
    print("prediccion score con SVR:",prediction)
    



    # probar salida en consola con
    # print(data)

    
