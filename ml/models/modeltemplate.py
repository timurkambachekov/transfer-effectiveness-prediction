from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, mean_squared_error as mse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelTemplate:
    
    def __init__(self, target, features, data) -> None:
        self.target = target
        self.features = features
        self.data = data.dropna(subset=[target, 'marketval_-1']).fillna(0)
        self.X, self.y = self.data[features], self.data[target]

        
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.__indices = dict(
            train = self.X_train.index.values,
            test = self.X_test.index.values,
        )
        
    def scale(self):
        self.scaler_X_train = StandardScaler()
        self.scaler_y_train = StandardScaler()
        self.scaler_X_test = StandardScaler()
        self.scaler_y_test = StandardScaler()
        self.X_train = self.scaler_X_train.fit_transform(self.X_train)
        self.y_train = self.scaler_y_train.fit_transform(self.y_train.values.reshape(-1, 1)).reshape(-1)
        self.X_test = self.scaler_X_test.fit_transform(self.X_test)
        self.y_test = self.scaler_y_test.fit_transform(self.y_test.values.reshape(-1, 1)).reshape(-1)
    
    def inverse_scale(self):
        self.X_test = pd.DataFrame(self.scaler_X_test.inverse_transform(self.X_test), columns=self.features, index=self.__indices['test'])
        self.X_train = pd.DataFrame(self.scaler_X_test.inverse_transform(self.X_train), columns=self.features, index=self.__indices['train'])
        self.y_train = pd.Series(self.scaler_y_train.inverse_transform(self.y_train.reshape(-1, 1)).reshape(-1), name=self.target, index=self.__indices['train'])
        self.y_test = pd.Series(self.scaler_y_test.inverse_transform(self.y_test.reshape(-1, 1)).reshape(-1), name=self.target, index=self.__indices['test'])
        self.y_pred = pd.Series(self.scaler_y_test.inverse_transform(self.y_pred.reshape(-1, 1)).reshape(-1), name=self.target+'_pred', index=self.__indices['test'])

        
    def mae(self):
        err = mae(self.y_test, self.y_pred)
        print(f'MAE = {err}')
        return err
    
    def mape(self):
        err = mape(self.y_test, self.y_pred)
        print(f'MAPE = {err}')
        return err
    
    def tune_hp(self):
        raise NotImplementedError
        
        
    def train(self):
        raise NotImplementedError
        
    def plot_predictions(self):        
        fig = make_subplots(rows=3, cols=1)
        
        x, y = self.y_test, abs(self.y_pred - self.y_test)
        fig.add_trace(go.Scatter(
            x = x, 
            y = y, 
            name='predictions', 
            mode='markers',
        ), 1, 1)
        fig.add_trace(go.Scatter(
            x = x, 
            y = np.poly1d(np.polyfit(x, y, 1))(x), 
            name='predictions trend', 
            mode='lines',
        ), 1, 1)
        
        x, y = self.y_test, abs(self.y_pred - self.y_test)
        fig.add_trace(go.Scatter(
            x = x, 
            y = y, 
            name='mae', 
            mode='markers',
        ), 2, 1)
        fig.add_trace(go.Scatter(
            x = x, 
            y = np.poly1d(np.polyfit(x, y, 1))(x), 
            name='mae trend', 
            mode='lines',
        ), 2, 1)
        
        x, y = self.y_test, abs(self.y_pred - self.y_test)/self.y_test
        fig.add_trace(go.Scatter(
            x = x,
            y = y, 
            name='mape', 
            mode='markers',
        ), 3, 1)
        fig.add_trace(go.Scatter(
            x = x, 
            y = np.polyval(np.polyfit(x, y, 1), x), 
            name='mape trend', 
            mode='lines',
        ), 3, 1)

        fig.update_layout(
            width = 1000,
            height = 800,
            title = 'Error distribution'
        )
        fig.show()
        
    def top_n_predictions(self, n, criteria='error', worst=False):
        player_info_cols = ['name', 'age', 'season', 'country_from', 'league_from', 'club_from',
                            'country_to', 'league_to', 'club_to', 'window', 'fee', 'loan']
        preds = pd.concat([self.data.loc[self.__indices['test'], player_info_cols], self.y_test, self.y_pred], axis=1)
        preds['error'] = abs(self.y_pred - self.y_test)
        preds = preds.sort_values('error', ascending = not worst)
        return preds.head(n)
        