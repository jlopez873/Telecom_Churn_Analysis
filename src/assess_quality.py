## Import libraries/packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (18,10)
plt.rcParams['figure.max_open_warning'] = False

## Import data
df = pd.read_csv('churn_raw_data.csv', index_col=0).reset_index().drop('index', axis=1)

## Rename survey columns
df.rename({
    'item1':'timely_response',
    'item2':'timely_fixes',
    'item3':'timely_replacements',
    'item4':'reliability',
    'item5':'options',
    'item6':'respectful_response',
    'item7':'courteous_exchange',
    'item8':'active_listening'
}, axis=1, inplace=True)

## Detect duplicates
print(df.duplicated().value_counts())

## Count all null values
print(df.isnull().sum())

## Assign zscores
for col in df.columns:
    if df[col].dtype == int or df[col].dtype == float:
        df['zscore_' + col] = zscore(df[col])
        
## Identify outliers
outliers = pd.DataFrame(columns=df.columns)
for col in df.columns:
    if 'zscore' in col:
        outliers = pd.concat([outliers, df.query('' + col + ' > 3 | ' + col + ' < -3')])

## Review unique outlier values
for col in df.columns:
    if 'zscore' in col:
        temp = df.query('' + col + ' > 3 | ' + col + ' < -3')[col[7:]].sort_values()
        print(temp.value_counts())
        print(col[7:] + ' length:', len(temp))
        print(col[7:] + ' percent of values:', (len(temp)/len(df))*100, end='\n\n')