import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.mixture import GaussianMixture

class TransfersDataset:
    
    def __init__(self, path) -> None:
        data = pd.read_csv(filepath_or_buffer=path, index_col=False).drop_duplicates().reset_index(drop=True)
        # data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        self.data = data
        self.gk_features = ['ga90','shotsa90','cs','saveratepct','xga90','prevgoals90','bpassesrcvd90','exits90','gkaerduels90']

        
    def encode_last_positions(self):
        positions = self.data['pos'].str.split(', ').explode().dropna().unique()
        t = self.data['pos'].str.split(', ')
        mapping = []
        for k in t.values:
            x = {}
            for i in k:
                for j in positions:
                    if i == j:
                        x[j] = 1
                    elif j not in x.keys(): 
                        x[j] = 0  
            mapping.append(x)
        self.data = self.data.drop(columns=['pos'])
        self.data = self.data.join(pd.DataFrame(mapping).add_prefix('pos'), self.data.index)
        
    def assign_clusters(self, X):
        gmm = GaussianMixture(n_components=4, covariance_type='diag', random_state=0)
        gmm.fit(X)
        self.labels = gmm.predict(X) 
        
    def filter_cluster(self, cluster):
        self.data = self.data.iloc[self.labels == cluster]
        
        
    def filter_postion(self, pos):
        position_mapping = dict(
            LWF='ATT',
            CF='ATT',
            LW='ATT',
            LCMF='MID',
            LCB='MID',
            RW='ATT',
            RAMF='MID',
            LB='DEF',
            RCMF='MID',
            RB='DEF',
            RDMF='MID',
            AMF='MID',
            LAMF='MID',
            LWB='DEF',
            RWF='ATT',
            RCB='DEF',
            GK='GK',
            RWB='DEF',
            LDMF='MID',
            DMF='MID',
            CB='DEF'
        )
        t = self.data['pos'].str.split(', ').explode().map(position_mapping).reset_index().drop_duplicates()
        t = t[t['pos'] == pos]['index']
        self.data = self.data.iloc[t]
        self.data = self.data.drop(columns=['pos'])
        if pos != 'GK':
            self.data = self.data.drop(columns=self.gk_features)
        
        
    def encode(self):
        self.data['season'] = OrdinalEncoder().fit_transform(self.data[['season']])
        self.data['window'] = self.data['window'].replace(to_replace={'summer': 1, 'winter': 0})
        self.data[self.data.filter(regex='foot').columns] = self.data.filter(regex='foot').replace(to_replace={'right': 1, 'left': 1, 'both':2})
        self.data[self.data.filter(regex='loanflg').columns] = self.data.filter(regex='loanflg').replace(to_replace={'no': 0, 'yes': 1})

    def drop(self):
        self.data = self.data.drop(columns=self.data.filter(regex='birthcountry|passpcountry|season_|pos_').columns)
        # self.data = self.data[['age', 'season', 'window', 'loan', 'club_from_elo', 'club_to_elo', 'league_from_elo', 'league_to_elo', 'marketval_0'] + \
        #                        self.data.columns[self.data.columns.str.contains('_-1')].tolist()]  
        self.data = self.data.drop(columns=['ycards', 'xga', 'xg', 'xa', 'shotsa', 'shots', 'rcards', 'prevgoals', # drop columns which have per 90 stat
                                           'npgoals', 'loanflg', 'loan', 'hdrgoals', 'goals','ga', 'assists'])
    