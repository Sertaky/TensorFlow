
### 1. DATA PREPROCESSING
"""

# 1 IMPORTING MODULES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sns.set(style="white")

import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('/content/housing.csv')
df.head()

"""### 2 DATA INSPECTING AND CLEANING"""

df.describe() #  inspecting numerical features

df.info() #  showing information about datatypes

df.isnull().sum() #  null values for all features

"""### Housing distribution on the map"""

plt.figure(figsize=(18,8))
fig = sns.scatterplot(df.longitude, df.latitude, hue=df['median_house_value'],
                      legend=False, palette='winter');

"""###  Data Imputation

* there's only 207 null values in total_bedrooms

* KNN is gonna be useful to inpute data to dataframe
"""

from sklearn.neighbors import KNeighborsRegressor


def input_knn(data_frame=df):
    
    """arguments: data_frame:pandas data frame - df default
         returns: data_frame with filled Nan values"""
    
    numeric_features = df.select_dtypes(include=[np.number]) #  only features with numeric values

    caterogical_features = df.select_dtypes(exclude=[np.number])  #  only features without numeric values

    nan_columns = numeric_features.columns[numeric_features.isna().any()].to_list() #  features with empty values (NaNs)
    
    no_nan_columns = numeric_features.columns.difference(nan_columns).values  #  features withouy empty values
    
    
    for column in nan_columns:
        imp_test = numeric_features[numeric_features[column].isna()] #  columns with null values 
        imp_train = numeric_features.dropna()
        model = KNeighborsRegressor(n_neighbors=5)
        knr = model.fit(imp_train[no_nan_columns], imp_train[column]) #  train model takes values from columns without null values
        numeric_features.loc[df[column].isna(), column] = knr.predict(imp_test[no_nan_columns]) #  KNR predicts replaces null values
        
    return pd.concat([numeric_features, caterogical_features], axis=1)

"""### Making copy of the updated data """

df_copy = df 

df=input_knn('total_bedrooms') #  imputing data

df.describe()

"""### 3 HISTOGRAMS

i'll use them to try to inspect :

* data distribution

* otuliners

* odd patterns

* scale of axis
"""

def plot_histogram(column_name, data_frame=df):
    
    """arguments: column_name:str - name of the column to be ploted
       returns: histogram object
    """
    
    #  in case when column is not in data frame
    if column_name not in data_frame.columns:
        raise ValueError(f'Chose correct column from data frame colums: {data_frame.columns}')
        
    fig = px.histogram(data_frame=data_frame.sort_values(by=column_name), x=column_name,
                       color_discrete_sequence=['blue'])
    
    
    fig.update_layout(font=dict(family='Lato', size=16), 
                      title=dict(text=f'<b>histogram - {column_name}<b>',
                                font=dict(size=24),
                                x=.5),
                     plot_bgcolor='lightblue',
                     paper_bgcolor='lightblue',
                     xaxis=dict(showgrid=False),
                     yaxis=dict(showgrid=False))
    
    fig.show()

numeric_columns = df.select_dtypes(np.number)
for column in numeric_columns:
    plot_histogram(column)

"""### Odd patterns and oustliners

* there's outliners at housing median age and median house value.

* housing median age has outliner at 52k, median house value at 500k

### Less noticible:

* total rooms column cointans outliners.

* total bedrooms column contains otuliners.

* population and household contains outliners 
"""

# Dropping outliners from population column
df.sort_values(by='population', ascending=False)

df = df.drop(labels = [15360, 9880])
plot_histogram('population')
df.sort_values(by='population', ascending=False)

"""### Correlation"""

df.corr().style.background_gradient()

def correlation_heatmap(data_frame=df):
    """arguments: data_frame:pandas DataFrame
       returns: correlation heatmap"""
    
    #  setting the context
    sns.set(context='paper')
    
    #  making correlation object and saving it into variable
    correlation = df.corr()
    
    #  creating heatmap figure object (paper) and ax object (the plot)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    #  generating color palettes
    cmap = sns.diverging_palette(220, 10, center='light', as_cmap=True)
    
    #  draw the heatmap
    heatmap = sns.heatmap(correlation, vmax=1,vmin=-1,center=0, square=False, annot=True, cmap=cmap,
                         lw=2, cbar=False)
    
    return heatmap

correlation_heatmap();

"""### median_house_value is quite strong correlated to median_income variable

### 4 Creating and Training Model

Linear Regression
"""

from sklearn.model_selection import train_test_split     #  splitting the data into train and test

X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.33,
                                                   random_state=42)

from sklearn.linear_model import LinearRegression

reg = LinearRegression() #  importing LinearRegression

reg.fit(X_train, y_train) #  fitting the train data frame and train feature to the LinearRegression

predictions=reg.predict(X_test)

print(f'actual: {y_test.mean()}')
print(f'predictions: {predictions.mean()}')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('MAE: ' + str(mean_absolute_error(y_test, predictions)))
print('MSE: ' + str(mean_squared_error(y_test, predictions)))
print('Score: '+ str(r2_score(y_test, predictions)))

"""2 Random Forest"""

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()

reg.fit(X_train, y_train) #  fitting train data and train feature

actual = y_test
predictions = reg.predict(X_test)
print(f'Actual mean: {np.mean(actual)}')
print(f'Predicted mean: {np.mean(predictions)}')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE: ' + str(mean_absolute_error(y_test, predictions)))
print('MSE: ' + str(mean_squared_error(y_test, predictions)))
print('Score: '+ str(r2_score(y_test, predictions)))

"""### 5.Conclusions

* Looks like RandomForestRegressor is the best model with score : 0.8129
"""

