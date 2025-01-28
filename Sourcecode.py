import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
 #loading the data from csv file to a pandas DataFrame
calories = pd.read_csv("/content/calories.csv")
#print the first 5 rows of the DataFrame
 calories.head()
 excercise_data = pd.read_csv("/content/exercise.csv")
 excercise_data.head()
 #Combining the two Dataframes
 calories_data = pd.concat([excercise_data, calories['Calories']], axis=1)
 calories_data.head()
 #Checking the number of rows and columns
 calories_data.shape
 #getting some information about the data
 calories_data.info()
#checking the missing values
calories_data.isnull().sum()
calories_data.describe()
sns.set()
#plotting the gender column in count plot
sns.countplot(calories_data['Gender'])
#finding the distribution of age column
sns.distplot(calories_data['Age'])
#finding the distribution of height column
sns.distplot(calories_data['Height'])
#finding the distribution of weight column
sns.distplot(calories_data['Weight'])
#finding correlation in the dataset
correlation = calories_data.corr()
#constructing a heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')
#converting the text data to numerical values
calories_data.replace({'Gender':{'male':0, 'female':1}}, inplace=True)
calories_data.head()
#separating features and target
x = calories_data.drop(columns=['User_ID','Calories'],axis=1)
y = calories_data['Calories']
print(x)
print(y)
#splitting the data into training data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
#model training
#XGBoost Regressor
#loading the model
model = XGBRegressor()
#training the model with X_train
model.fit(X_train, Y_train)
#prediction of test data
test_data_prediction = model.predict(x_test)
print(test_data_prediction)
#mean absolute error
mae = metrics.mean_absolute_error(y_test, test_data_prediction)
print("Mean Absolute Error = ", mae)
