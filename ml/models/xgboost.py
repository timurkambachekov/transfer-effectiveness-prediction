from models.modeltemplate import ModelTemplate

from hyperopt import hp, tpe, Trials, fmin
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from IPython.display import clear_output


class Xgboost(ModelTemplate):
    
    def __init__(self, target, features, data) -> None:
        super().__init__(target, features, data)
        
    def tune_hp(self):
        space = {
            'max_depth': hp.choice('max_depth', range(31, 100)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'n_estimators': hp.choice('n_estimators', range(100, 1500, 100)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'gamma': hp.uniform('gamma', 0.0, 0.5),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
        }

        # Define objective function to minimize (in this case, mean squared error)
        def objective(params):
            clf = xgb.XGBRegressor(**params)
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            return mse(self.y_test, y_pred)

        # Perform hyperparameter optimization
        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=50,  # Number of iterations
                    trials=trials)

        print("Best parameters:", best)
        self.best_params = best
        
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
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        bst = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtest, 'eval')], early_stopping_rounds=20)
        self.y_pred = bst.predict(dtest)
        clear_output()