import pandas as pd


class ClassToInt:
    '''
    transforms object or category pandas columns to integer.  
    '''
    def __init__(self):
        self.columns = {}
        self.fitted = False
        
    def fit(self, df):
        '''
        ARGUMENTS
        ---------------
        df: (pd.DataFrame) 
        '''
        for col in df.columns:
            if (df[col].dtype.name in ['object','category']) or ('int' in df[col].dtype.name):
                print(col)
                if df[col].isnull().sum() >0:
                    raise Exception(f'ERROR: {col} has NA values.')
                 
                self.columns[col] = {}
                self.columns[col]['names'] = {name:i for i, name in enumerate(df[col].unique())}
                self.columns[col]['category'] = df[col].dtype.name=='category'
        self.fitted = True
    
    def transform(self, df):
        if self.fitted == False:
            raise Exception(f'ERROR: Cannot transform before calling `fit` method.')
        
        for col in self.columns:
            df[col] = df[col].map(self.columns[col]['names'])
            if self.columns[col]['category']:
                df[col] = df[col].astype('int')
        return df
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
