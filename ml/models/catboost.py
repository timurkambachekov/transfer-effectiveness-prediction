from models.modeltemplate import ModelTemplate

import optuna 
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as mse
from IPython.display import clear_output


class CatBoost(ModelTemplate):
    best_params = {
        'iterations': 4441,
        'learning_rate': 0.035725014467313174,
        'depth': 5,
        'subsample': 0.7756176221984215,
        'colsample_bylevel': 0.9114327325766978,
        'min_data_in_leaf': 4,
        'random_seed': 42
    }
    default_params = {
        'iterations': 300,
        'random_seed': 42,
        'silent': True
    }
    
    def __init__(self, data, features=None, feature_selection=False, full_feature_set=False) -> None:
        super().__init__(data, features, full_feature_set)
        if feature_selection:
            self.model = CatBoostRegressor(**self.default_params)
        else:
            self.model = CatBoostRegressor(**self.best_params)
        
    def tune_hp(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 200, 5000),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
                "depth": trial.suggest_int("depth", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            }

            cb = CatBoostRegressor(**params, silent=True)
            cb.fit(self.X_train, self.y_train, eval_set=(self.X_test, self.y_test), early_stopping_rounds=50, silent=True)
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
        m.fit(self.X_train, self.y_train, silent=True)
        self.y_pred = m.predict(self.X_test)
        self.model = m
        
    def feature_importance(self):
        fi = pd.DataFrame(self.model.get_score(importance_type='gain').items(), columns=['feature', 'importance'])
        fi = fi.sort_values('importance', ascending=False)
        return fi