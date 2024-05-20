from models.modeltemplate import ModelTemplate

from hyperopt import hp, tpe, Trials, fmin
import xgboost as xgb
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from IPython.display import clear_output


class Xgboost(ModelTemplate):
    default_params = {
        'verbosity': 0
    }
    best_params = {
        'max_depth': 93,
        'learning_rate': 0.09928429023652342,
        'n_estimators': 473,
        'subsample': 0.9531552354359136,
        'colsample_bytree': 0.7346644282110879,
        'gamma': 0.2910912709165702,
        'reg_alpha': 0.08111455203904866,
        'reg_lambda': 0.7428293688241424,
        'verbosity': 0
    } 
    
    def __init__(self, data, features=None, feature_selection=False, full_feature_set=False) -> None:
        super().__init__(data, features, full_feature_set)
        if feature_selection:
            self.model = xgb.XGBRegressor(**self.default_params)
        elif self.best_params:
            self.model = xgb.XGBRegressor(**self.best_params)
        else:
            self.model = xgb.XGBRegressor()
            
            
    def tune_hp(self):
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'verbosity': 0
            }
            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            dtest = xgb.DMatrix(self.X_test, label=self.y_test)
            bst = xgb.train(params, dtrain)
            preds = bst.predict(dtest)
            rmse = mse(self.y_test, preds, squared=False)  # RMSE
            return rmse
        # Create a study object and optimize the objective function
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        print("Best parameters:", study.best_params)
        self.best_params = study.best_params
        
    def train(
        self, 
        params = {
            'colsample_bytree': 0.7977681279340311, 
            'gamma': 0.4849057724692036, 
            'learning_rate': 0.2438380375208144, 
            'max_depth': 57, 
            'n_estimators': 12, 
            'reg_alpha': 0.05504741360493881, 
            'reg_lambda': 0.5229731131689913, 
            'subsample': 0.7419233298225673}
        ):
        if self.best_params:
            params = self.best_params
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.model = xgb.train(params, dtrain, num_boost_round=10000, evals=[(dtest, 'eval')], early_stopping_rounds=50, verbose_eval=0)
        self.y_pred = self.model.predict(dtest)
        
    def feature_importance(self, log_name):
        fi = pd.DataFrame(self.model.get_score(importance_type='gain').items(), columns=['feature', 'importance'])
        fi = fi.sort_values('importance', ascending=False)
        fi.to_csv(f'feature_importance_{log_name}.csv', index=False)
        return fi