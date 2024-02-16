import pandas as pd
import joblib

class Utils:
    
    # funcioncargar data con funcion load_from_csv
    def load_from_csv(self, path):
        return pd.read_csv(path)

    # funcion para separar features y target
    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1) # quitar columna y dejar solo features
        y = dataset[y] # definir target
        return X,y
    

    # funcion exportar modelo, clf clasificador
    def model_export(self, clf, score):
        print("score model training:", score)
        joblib.dump(clf, './models/best_model.pkl')