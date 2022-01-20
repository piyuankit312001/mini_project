import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

car_df = pd.read_csv("Car details v3.csv")
bike_df = pd.read_csv("Bikes Best Buy.csv")

car_df['Year_Old'] = 2020 - car_df['year']
car_df.drop(["owner", "seats", "seller_type", "name", "max_power", "torque", "year"], axis = 1, inplace = True)

car_df.drop_duplicates(inplace=True)
car_df.dropna(subset = ['mileage'], inplace = True)

def find_number(text):
    num = re.findall(r'[0-9]+',text)
    return " ".join(num)

car_df['engine']=car_df['engine'].apply(lambda x: find_number(x))
car_df['mileage']=car_df['mileage'].apply(lambda x: find_number(x))
car_df['mileage'] = car_df['mileage'].str.replace(' ', '.')

car_df[['engine']] = car_df[['engine']].astype("int")
car_df[['mileage']] = car_df[['mileage']].astype("float")

a = pd.get_dummies(car_df['transmission'], drop_first=True)
car_df = car_df.join(a)
car_df['fuel'].replace('LPG', 'CNG', inplace = True)
a = pd.get_dummies(car_df['fuel'])
car_df = car_df.join(a)
car_df.drop(["fuel", "transmission"], axis = 1, inplace = True)
car_df["E"] = 0

car_df.rename({'selling_price':'Price (INR)', 'mileage':'Mileage', 'km_driven':'KM_Driven', 'engine':'Engine (cc)', 'Diesel':'D', 'CNG':'G', 'Petrol':'P'}, axis = 1, inplace = True)
titles = ['Price (INR)', 'Engine (cc)', 'Year_Old', 'KM_Driven', 'Manual', 'D', 'G', 'P', 'E', 'Mileage']
car_df = car_df[titles]

a = pd.get_dummies(bike_df['Fuel'])
bike_df = bike_df.join(a)
bike_df["Manual"] = 1
bike_df["G"] = 0
bike_df["KM_Driven"] = 0
bike_df["Year_Old"] = 0

bike_df.drop(["Bike Name", "Company", "Fuel"], axis = 1, inplace = True)
bike_df.rename({'Price(INR)':'Price (INR)', 'Milage (kM/L)':'Mileage', 'Tank size (cc) ':'Engine (cc)'}, axis = 1, inplace =True)
titles = ['Price (INR)', 'Engine (cc)', 'Year_Old', 'KM_Driven', 'Manual', 'D', 'G', 'P', 'E', 'Mileage']
bike_df = bike_df[titles]

convert_dict = {'Price (INR)': int,
                'Engine (cc)': int,
                'Year_Old' : int,
                'KM_Driven': int,
                'Manual': int,
                'D': int,
                'G': int,
                'P': int,
                'E': int,
                'Mileage': float
               }

bike_df = bike_df.astype(convert_dict)
car_df = car_df.astype(convert_dict)
df = car_df.append(bike_df)

df = df[df['Mileage']!=0]
df = df[df['Engine (cc)']!=0]
df = df[df['E']==0]

df.drop(['E', 'G'], axis = 1, inplace = True)

y = df.pop('Mileage')
X = df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor()
# Hyperparameters
n_estimators = [100,200,300,500]
max_features = ['auto', 'sqrt']
max_depth = [5, 10, 15, 20]

from sklearn.model_selection import RandomizedSearchCV
grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth
}

clf_cv = RandomizedSearchCV(estimator=clf, param_distributions=grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42)

clf_cv.fit(X_train, y_train)

pickle.dump(clf_cv, open('model.pkl','wb'))