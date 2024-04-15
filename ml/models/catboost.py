from models.modeltemplate import ModelTemplate

import optuna 
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as mse
from IPython.display import clear_output


class CatBoost(ModelTemplate):
    best_params = {
        "iterations": 1000,
        "learning_rate": 0.01,
        "depth": 5,
        "subsample": 1,
        "colsample_bylevel": 1,
        "min_data_in_leaf": 10,
    }
    
    def __init__(self, target, features, data) -> None:
        super().__init__(target, features, data)
        
    def tune_hp(self):
        def objective(trial):
            params = {
                "iterations": 1000,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "depth": trial.suggest_int("depth", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            }

            cb = CatBoostRegressor(**params, silent=True)
            cb.fit(self.X_train, self.y_train, eval_set=(self.X_test, self.y_test), early_stopping_rounds=50, verbose=False)
            val_score = cb.best_score_['validation']['RMSE']
            return val_score

        # Create a study object and optimize the objective function
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        self.best_params = study.best_params
        print("Best parameters:", study.best_params)
        
    def train(
        self):
        m = CatBoostRegressor(**self.best_params)
        m.fit(self.X_train, self.y_train)
        self.y_pred = m.predict(self.X_test)
        clear_output()