# Movie Dataset 
In this dataset I would mainly focus upon the Linear Regression portion of Machine Learning. 
Here, I have a movies dataset which I work upon in this project to perform basic exploratory data analysis , data visualisation and predications.This Dataset contains information like : - Rank, Release Date, Movie Title, Production_Budget, Worldwide_Gross, Domestic_Gross. 

## Project Source 
The 100 Numbers.com 

## Project Outcomes:
by the end of this project we would have sucessfully answered a few questions :

### *The Imports*
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

#### Reading the data
```
movie = pd.read_csv("2.2 cost_revenue_dirty.csv.csv")
movie.head()
```

#### Reading the data
```
movie = pd.read_csv("2.2 cost_revenue_dirty.csv.csv")
movie.head()
```

### *Data Cleaning*

#### Removing Data Cells with 0 Gross Produce
```
movie.drop(movie[(movie["Worldwide Gross ($)"] == '$0') & (movie["Domestic Gross ($)"] == '$0')].index,inplace=True)
movie.tail()
```

#### Changing the column names for Production Budget, Worldwide Gross and Domestic Gross
```
movie.rename(columns={'Production Budget ($)':'Production_Budget','Worldwide Gross ($)':'Worldwide_Gross','Domestic Gross ($)':'Domestic_Gross'},inplace=True)
movie.head(2)
```
#### Changing the data types for gross and production budget
```
movie['Production_Budget'] = pd.to_numeric(movie['Production_Budget'])
movie['Worldwide_Gross'] = pd.to_numeric(movie['Worldwide_Gross'])
movie['Domestic_Gross'] = pd.to_numeric(movie['Domestic_Gross'])
movie['Release Date'] = pd.to_datetime(movie['Release Date'])
```

### *Exploratory Data Analysis*

#### Basic Description of our dataset
```
movie.describe()
```

#### Information about our dataset
```
movie.info()
```

#### How many movies were released during 2015 and 2019
```
movie[(movie['Release Date'] > '2015-01-01') & (movie['Release Date'] < '2019-12-31')]
```

#### Movies with production budget greater than 11000000 and released before 2015
```
movie[(movie['Production_Budget'] > 11000000) & (movie['Release Date'] < '2015-01-01')]
```

### *Visualisation*

#### Plot a scatterplot for production budget vs worldwide gross
```
plt.figure(figsize=(10,6))
sns.scatterplot(x='Production_Budget',y='Worldwide_Gross',data=movie,s=70)
```

#### Plot a Graph for worldwide gross vs domestic gross
```
plt.figure(figsize=(10,6))
plt.figure(figsize=(10,6))
sns.scatterplot(x='Worldwide_Gross',y='Domestic_Gross',data=movie,s=80)
```

#### Add a countplot for number of movies released during 2010 and 2015
```
movie['Year'] = movie['Release Date'].dt.year
movie[(movie['Year'] > 2010 ) & (movie['Year'] < 2015)]

plt.figure(figsize=(10,6))
sns.countplot(x='Year',data=movie[(movie['Year'] > 2000 ) & (movie['Year'] < 2015)],color='orange',hatch='/',lw=3,ec='white')
```

### *Data Predictions*
```
from sklearn.model_selection import train_test_split
```
#### Divide the data into Training and Testing sets
```
movie.columns

X = movie[['Production_Budget','Worldwide_Gross']]
y = movie['Domestic_Gross']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)
```

#### Training the model using linear regression
```
from sklearn.linear_model import LinearRegression
```
#### Create an object of LinearRegression()
```
lm =LinearRegression()
```
#### Train/fit your model
```
lm.fit(X_train, y_train)
```

#### Print out the coefficients of the model
```
lm.coef_
```

#### Predicting Test Data

#### Now that we have fit our model lets evaluate its performance by predicting off the test values
```
predictions = lm.predict(X_test)
```

#### Plotting a scatterplot for real test values vs the predicted values
```
plt.figure(figsize=(14,4))
plt.ylabel('True Values')
sns.regplot(x=y_test,y=predictions,fit_reg=True,ci=25,marker='o',scatter_kws={'color':'green','s':90,'ec':'white'})
plt.grid()
```

### * Evaluate the model*

#### Calculate the mean absolute error, mean squared error, and the root mean square error
```
from sklearn import metrics

print("MAE :",metrics.mean_absolute_error(y_test,predictions))
print("MSE :",metrics.mean_squared_error(y_test,predictions))
print("RMSE :",np.sqrt(metrics.mean_squared_error(y_test,predictions)))
```

#### Explained Variance Score
```
metrics.explained_variance_score(y_test,predictions)
```

#### Plot a histogram of the residuals to make sure it looks normally distributed
```
sns.displot(y_test-predictions,bins=50,kde=True,lw=2,edgecolor='white',color='green',aspect=2)
```

#### We need an answer to the question of whether should one invest there money in movie or not
```
pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
```

## Project Setup:
To clone this repository you need to have Python compiler installed on your system alongside pandas and seaborn libraries. I would rather suggest that you download jupyter notebook if you've not already.

To access all of the files I recommend you fork this repo and then clone it locally. Instructions on how to do this can be found here: https://help.github.com/en/github/getting-started-with-github/fork-a-repo

The other option is to click the green "clone or download" button and then click "Download ZIP". You then should extract all of the files to the location you want to edit your code.

Installing Jupyter Notebook: https://jupyter.readthedocs.io/en/latest/install.html<br>
Installing Pandas library: https://pandas.pydata.org/pandas-docs/stable/install.html












