from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, mean_squared_error as mse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
from IPython.display import clear_output, display

class ModelTemplate:
    target = 'marketval_0'
    
    def __init__(self, data, features=None, full_feature_set=False) -> None:
        to_drop = ['marketval_0', 'name', 'market_value', 'country_from', 'league_from', 'club_from', 'country_to', 'league_to', 'club_to', 'playerid']
        selected_features = ['age', 'season', 'window', 'fee', 'club_from_elo', 'club_to_elo', 'league_from_elo', 'league_to_elo', 
                             'marketval', 'matchesplayed', 'minsplayed', 'foot', 'height', 'weight']
        if features != None:
            if full_feature_set:
                self.features = features
            else:
                self.features = selected_features + features
        else:
            self.features = data.drop(columns=to_drop).columns.tolist()
        self.data = data.fillna(0)
        self.model = None
        self.X, self.y = self.data[self.features], self.data[self.target]

     
    def league_adjust(self):
        selected_features = ['age', 'season', 'window', 'fee', 'club_from_elo', 'club_to_elo', 'league_from_elo', 'league_to_elo', 
                             'marketval', 'matchesplayed', 'minsplayed', 'foot', 'height', 'weight']
        
        cols = self.data[self.features].drop(columns=selected_features).columns
        self.data.loc[:,cols] *= self.data.club_from_elo / self.data.club_to_elo
        
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
        self.X_train = pd.DataFrame(self.scaler_X_train.fit_transform(self.X_train), columns=self.features)
        self.y_train = pd.Series(self.scaler_y_train.fit_transform(self.y_train.values.reshape(-1, 1)).reshape(-1), name=self.target)
        self.X_test = pd.DataFrame(self.scaler_X_test.fit_transform(self.X_test), columns=self.features)
        self.y_test = pd.Series(self.scaler_y_test.fit_transform(self.y_test.values.reshape(-1, 1)).reshape(-1), name=self.target)
    
    def inverse_scale(self):
        self.X_test = pd.DataFrame(self.scaler_X_test.inverse_transform(self.X_test), columns=self.features, index=self.__indices['test'])
        self.X_train = pd.DataFrame(self.scaler_X_test.inverse_transform(self.X_train), columns=self.features, index=self.__indices['train'])
        self.y_train = pd.Series(self.scaler_y_train.inverse_transform(self.y_train.to_frame()).reshape(-1), name=self.target, index=self.__indices['train'])
        self.y_test = pd.Series(self.scaler_y_test.inverse_transform(self.y_test.to_frame()).reshape(-1), name=self.target, index=self.__indices['test'])
        self.y_pred = pd.Series(self.scaler_y_test.inverse_transform(self.y_pred.reshape(1, -1)).reshape(-1), name=self.target+'_pred', index=self.__indices['test'])

        
    def score(self, thresh=3):
        res = pd.concat([self.y_test, self.y_pred], axis=1).set_axis(['true', 'pred'], axis=1)
        err = dict(
            mae = mae(res.true, res.pred),
            mape = mape(res.true, res.pred),
            mae_below_thresh = mae(res[res.true < thresh]['true'], res[res.true < thresh]['pred']),
            mape_above_thresh = mape(res[res.true >= thresh]['true'], res[res.true >= thresh]['pred'])
        )
        print(f'thresh = {thresh}')
        for metric, score in err.items():
            print(f'{metric}: {score}')
        
        err['features'] = [self.features]
        return pd.DataFrame.from_dict(err)
    
    def mape(self):
        err = mape(self.y_test, self.y_pred)
        print(f'MAPE = {err}')
        return err
    
    def tune_hp(self):
        raise NotImplementedError
        
        
    def train(self):
        raise NotImplementedError
    
    def feature_importance(self):
        raise NotImplementedError
    
    def forward_feature_selection(self, n, log_name):
        selected_features = ['age', 'season', 'window', 'fee', 'club_from_elo', 'club_to_elo', 'league_from_elo', 'league_to_elo', 
                             'marketval', 'matchesplayed', 'minsplayed', 'foot', 'height', 'weight']
        rest_features = self.X_train.drop(columns=selected_features).columns.tolist()
        current_best_score = -np.inf
        logs = []
        # Define stopping criterion (e.g., maximum number of features)
        max_features = min(n, self.X_train.shape[1])
        for i in tqdm(range(max_features)):
            rest_features = self.X_train.drop(columns=selected_features).columns.tolist()
            best_feature = None
            best_score = -np.inf
            
            for feature in tqdm(rest_features, leave=False):
                features_to_try = selected_features + [feature]
                cv_scores = cross_val_score(self.model, self.X_train[features_to_try], self.y_train, cv=3, scoring='neg_mean_squared_error')
                score = np.mean(cv_scores)
                    
                if score > best_score:
                    best_score = score
                    best_feature = feature
                        
            selected_features.append(best_feature)
            if best_score > current_best_score:
                current_best_score = best_score
            # else:
            #     break  # Stop if adding more features does not improve performance significantly
            logs.append([", ".join(selected_features), -best_score])
            print(f'Iteration {i+1}', f'Features: {", ".join(selected_features)}', f'Score: {-best_score}', sep='\n')
            
        pd.DataFrame(logs, columns=['Features', 'MSE']).to_csv(f'{log_name}.csv')
        print(f'Best Score: {current_best_score}')
        return selected_features
        
    def plot_predictions(self):        
        fig = make_subplots(rows=3, cols=1)
        
        x, y = self.y_test, self.y_pred
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
            title = 'Model results',
            xaxis1 = dict(title_text='Market value, Euro, mln.'),
            xaxis2 = dict(title_text='Market value, Euro, mln.'),
            xaxis3 = dict(title_text='Market value, Euro, mln.'),
            yaxis1 = dict(title_text='Euro, mln.'),
            yaxis2 = dict(title_text='Euro, mln.'),
            yaxis3 = dict(title_text='PPT'),
        )
        fig.show()
        
    def top_n_predictions(self, n, criteria='error', worst=False):
        player_info_cols = ['name', 'age', 'season', 'country_from', 'league_from', 'club_from',
                            'country_to', 'league_to', 'club_to', 'window', 'marketval', 'marketval_0', 'fee']
        preds = pd.concat([self.data.loc[self.__indices['test'], player_info_cols], self.y_test, self.y_pred], axis=1)
        preds.season = preds.season.map({
            0: '19/20',
            1: '20/21',
            2: '21/22',
            3: '22/23',
            4: '23/24'
        })
        preds['error'] = abs(self.y_pred - self.y_test)
        preds['error_pct'] = abs(self.y_pred - self.y_test) / self.y_test
        self.predictions = preds
        preds = preds.sort_values(criteria, ascending = not worst)
        return preds.head(n)
    
    
    def calculate_effectiveness(self):
        inflation = pd.read_csv('/Users/timurkambachekov/вышка/4 курс/вкр/project/prepped/inflation.csv')
        age_coef = pd.read_csv('/Users/timurkambachekov/вышка/4 курс/вкр/project/prepped/age_coeffiecient.csv')
        
        inflation.inflation = inflation.inflation.shift(-1)
        self.predictions = self.predictions.merge(age_coef, how='left', on='age').merge(inflation, how='left', on='season')
        def effectiveness(row, marketval_0):
            return ((row['age_coef'] * (1 + row['inflation']) ** -1 * row[marketval_0] - row['marketval'] + (row['marketval'] - row['fee']))) / (2 * row['marketval'])  
        
        self.predictions['eff_true'] = self.predictions.apply(lambda x: effectiveness(x,'marketval_0'), axis=1).iloc[:, 1:]
        self.predictions['eff_pred'] = self.predictions.apply(lambda x: effectiveness(x,'marketval_0_pred'), axis=1)
