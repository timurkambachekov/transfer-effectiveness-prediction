import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.mixture import GaussianMixture

class TransfersDataset:
    
    def __init__(self, path) -> None:
        data = pd.read_csv(filepath_or_buffer=path, index_col=False).drop_duplicates().reset_index(drop=True)
        # data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        self.data = data
        
        
    def encode_last_positions(self):
        positions = self.data.filter(regex='pos_-', axis=1).apply(lambda col: col.unique()).explode().str.split(', ').explode().dropna().unique()
        t = self.data.filter(regex='pos_-', axis=1).apply(lambda row: row.str.cat(), axis=1).str.split(', ')
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
        t = self.data['pos_-1'].str.split(', ').explode().map(position_mapping).reset_index().drop_duplicates()
        t = t[t['pos_-1'] == pos]['index']
        self.data = self.data.iloc[t]
        
    def encode(self):
        self.data['season'] = OrdinalEncoder().fit_transform(self.data[['season']])
        self.data['window'] = self.data['window'].replace(to_replace={'summer': 1, 'winter': 0})
        self.data['loan'] = self.data['loan'].replace(to_replace={True: 1, False: 0})
        self.data['fee'] *= 10**6
        self.data[self.data.filter(regex='foot').columns] = self.data.filter(regex='foot').replace(to_replace={'right': 1, 'left': 1, 'both':2})
        self.data[self.data.filter(regex='loanflg').columns] = self.data.filter(regex='loanflg').replace(to_replace={'no': 0, 'yes': 1})

    def drop(self):
        self.data = self.data.drop(columns=self.data.filter(regex='birthcountry|passpcountry|season_|pos_').columns)    
    