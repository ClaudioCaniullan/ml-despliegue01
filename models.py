import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils

class Models:
    
    # modelos utilizar
    def __init__(self):
        self.reg = {
            'SVR' : SVR(),
        }
        
        # parametros a optimizar
        self.params = {
           'SVR' : {
               'kernel' : ['linear', 'poly', 'rbf'],
               'gamma' : ['auto', 'scale'],
               'C' : [1,5,10]
        }

    def grid_training(self, X,y):

        best_score = 999
        best_model = None
        
        # recorrer modelos
        for name, reg in self.reg.items():
            
            # recorrer paramatros, cv cross validation size=3
            grid_reg = GridSearchCV(reg, self.params[name], cv=3).fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)
            
            # comparar seleccionar el mejor score y modelo 
            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_
        
        # exportar mejor modelo
        utils = Utils()
        utils.model_export(best_model, best_score)

