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

## Impute missing values with median (skewed/bi-modally distributed variables)
cols = ['Children', 'Income', 'Tenure', 'Bandwidth_GB_Year']
for col in cols:
    print(col + ': ', df[col].median())
    df[col].fillna(df[col].median(), inplace=True)
    print(col, ': ', df[col].median())

## Impute missing values with mean (uniformally distributed variables)
print(col, ': ', df.Age.mean())
df.Age.fillna(df.Age.mean(), inplace=True)
print(col, ': ', df.Age.mean())

## Impute missing values with mode (categorical variables)
cols = ['Techie', 'Phone', 'TechSupport']
for col in cols:
    print(col, ': ', df[col].mode()[0])
    df[col].fillna(df[col].mode()[0], inplace=True)
    print(col, ': ', df[col].mode()[0])

## Verify all values were imputed
print(df.isnull().sum())
        
## Remove z-score
for col in df.columns:
    if 'zscore' in col:
        df.drop(col, axis=1, inplace=True)
        
## Store data
df.to_csv('cleaned_data.csv')