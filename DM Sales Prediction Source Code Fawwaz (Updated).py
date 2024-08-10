import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#Read the data
df_train = pd.read_csv("C:\\Users\\fnn09\\Downloads\\Train(1).csv")


#see dataset information
df_train.info()

# filling the missing values in "Item_weight column" with "Mean" value
df_train['Item_Weight'].mean()
df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(), inplace=True)

# filling the missing values in "Outlet_Size" column with Mode
#Here we take Outlet_Size column & Outlet_Type column since they are correlated
mode_of_Outlet_size = df_train.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
print(mode_of_Outlet_size)

miss_values = df_train['Outlet_Size'].isnull()
df_train.loc[miss_values, 'Outlet_Size'] = df_train.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

# Item_Weight distribution
sns.distplot(df_train['Item_Weight'], color='purple')
plt.show()

# Item Visibility distribution
sns.distplot(df_train['Item_Visibility'], color='purple')
plt.show()

# Item MRP distribution
sns.distplot(df_train['Item_MRP'], color='purple')
plt.show()

# Item_Outlet_Sales distribution
sns.distplot(df_train['Item_Outlet_Sales'], color='purple')
plt.show()

# Outlet_Establishment_Year column
sns.countplot(x='Outlet_Establishment_Year', data=df_train)
plt.show()

# Item_Fat_Content column
sns.countplot(x='Item_Fat_Content', data=df_train)
plt.show()

# Item_Type column
plt.figure(figsize=(25,7))
sns.countplot(x='Item_Type', data=df_train)
plt.show()

# Outlet_Size column
sns.countplot(x='Outlet_Size', data=df_train)
plt.show()


df_train.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)

#LABEL ENCODING
encoder = LabelEncoder()
df_train['Item_Identifier'] = encoder.fit_transform(df_train['Item_Identifier'])
df_train['Item_Fat_Content'] = encoder.fit_transform(df_train['Item_Fat_Content'])
df_train['Item_Type'] = encoder.fit_transform(df_train['Item_Type'])
df_train['Outlet_Identifier'] = encoder.fit_transform(df_train['Outlet_Identifier'])
df_train['Outlet_Size'] = encoder.fit_transform(df_train['Outlet_Size'])
df_train['Outlet_Location_Type'] = encoder.fit_transform(df_train['Outlet_Location_Type'])
df_train['Outlet_Type'] = encoder.fit_transform(df_train['Outlet_Type'])
df_train.head()

#Let's have all the features in X & target in Y
X = df_train.drop(columns='Item_Outlet_Sales', axis=1)
Y = df_train['Item_Outlet_Sales']

#SPLITTING THE DATA INTO TRAINING DATA & TESTING DATA
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

regressor = XGBRegressor()
#fit the model
#Training data is in X_train and the corresponding price value is in the Y_train
regressor.fit(X_train, Y_train)

#PREDICTION OF THE DATA
sales_data_prediction = regressor.predict(X_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_train, sales_data_prediction)
print('Mean Absolute Error (MAE):', mae)

# Mean Squared Error (MSE)
mse = mean_squared_error(Y_train, sales_data_prediction)
print('Mean Squared Error (MSE):', mse)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean Squared Error (RMSE):', rmse)

# In order to check the performance of the model we find the R squared Value
r2_sales = metrics.r2_score(Y_train, sales_data_prediction)
print('R Squared value = ', r2_sales)


input_data = (156, 9.300, 0, 0.016047, 4, 249.8092, 9, 1999,1, 0, 1)
print("The sales for the first product which is FDA15 in the dataset is predicted as ", sales_data_prediction[0])


















