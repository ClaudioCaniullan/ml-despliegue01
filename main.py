# importar utilitarios en la carpeta raiz
from utils import Utils
from models import Models

if __name__ == "__main__":
    
    # importar clases de los utilitarios
    utils = Utils()
    models = Models()
    
    # cargar data desde carpeta in
    data = utils.load_from_csv('./in/felicidad.csv')

    # definir features y target
    X, y = utils.features_target(data, ['score','rank', 'country'],['score'])
    
    # entrenar modelo
    models.grid_training(X,y)
    


    # probar salida en con sola con
    # print(data)

    
