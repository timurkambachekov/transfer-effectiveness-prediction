from models.modeltemplate import ModelTemplate

import optuna 
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as mse
from IPython.display import clear_output


class LightGBM(ModelTemplate):
    best_params = None
    default_params = {
        'objective': 'regression',  # Regression task
        'metric': 'rmse',            # Root Mean Squared Error
        'num_leaves': 31,            # Number of leaves in each tree
        'learning_rate': 0.01,       # Learning rate
        # 'feature_fraction': 0.9,     # Percentage of features to consider in each iteration
        # 'bagging_fraction': 0.8,     # Percentage of data to sample in each iteration (bagging)
        # 'bagging_freq': 5,           # Frequency for bagging
        # 'num_boost_round': 100,      # Number of boosting rounds
        'verbose': -1                 # No output during training
    }
    
    def __init__(self, data, features=None, feature_selection=False, full_feature_set=False) -> None:
        super().__init__(data, features, full_feature_set)
        if feature_selection:
            self.model = lgb.LGBMRegressor(**self.default_params)
        elif self.best_params:
            self.model = lgb.LGBMRegressor(**self.best_params)
        else:
            self.model = lgb.LGBMRegressor()
        
    def tune_hp(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            param = {
                'objective': 'regression',  # Regression objective
                'metric': 'rmse',           # Root Mean Squared Error for evaluation
                'verbose': -1,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 2, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0),
            }
            
            dtrain = lgb.Dataset(self.X_train, label=self.y_train, params={'verbose': -1,})
            dtest = lgb.Dataset(self.X_test, label=self.y_test, params={'verbose': -1,})

            bst = lgb.train(param, dtrain, verbose_eval=False)
            preds = bst.predict(self.X_test)
            rmse = mse(self.y_test, preds, squared=False)  # RMSE
            return rmse

        # Create a study object and optimize the objective function
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        self.best_params = study.best_params
        print("Best parameters:", study.best_params)
        
    def train(self):
        dtrain = lgb.Dataset(self.X_train, label=self.y_train, params={'verbose': -1,})
        dtest = lgb.Dataset(self.X_test, label=self.y_test, params={'verbose': -1,})
        m = lgb.train(self.best_params, dtrain, valid_sets=[dtest], verbose_eval=False)
        self.y_pred = m.predict(self.X_test)
        self.model = m
        
    def feature_importance(self):
        fi = pd.DataFrame(self.model.get_score(importance_type='gain').items(), columns=['feature', 'importance'])
        fi = fi.sort_values('importance', ascending=False)
        return fi