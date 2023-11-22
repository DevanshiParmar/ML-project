# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:33:52 2021

@author: devanshi
"""

#import necessary libraries
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("C:/Users/devan/Documents/BEAD Internship/final/c_events.csv")

data["datetime"]=data["datetime"].apply(lambda x: x.replace("-",""))
data["datetime"]=data["datetime"].apply(lambda x: x.replace(":",""))
data["datetime"]=data["datetime"].apply(lambda x: x.replace(" ",""))

print(data)
data.shape

#reading data into panda dataframe
wdata=pd.read_csv("C:/Users/devan/Documents/BEAD Internship/final/c_weather.csv")

wdata["datetime"]=wdata["datetime"].apply(lambda x: x.replace("-",""))
wdata["datetime"]=wdata["datetime"].apply(lambda x: x.replace(":",""))
wdata["datetime"]=wdata["datetime"].apply(lambda x: x.replace(" ",""))

print(wdata)
wdata.shape

data.drop(['propertyid', 'propfloorid','citykey'], axis=1, inplace=True)
pd.set_option("display.max.columns", None)
data.columns
data.shape
print(data)


out=wdata['outdoortemperature']
type(out)

dt=wdata['datetime']

#creating new column in main dataset
data['outdoor_temp']=""

da=data['outdoor_temp']

dtime=data["datetime"]

i=0
t=0
j=0

#adding outdoor temp column to main dataset
for i in range(len(dtime)):
    j=0
    for j in range(len(dt)):
        if (dtime[i] == dt[j]):
            da[i]=out[j]
            #print(da[i])
            
print(data)    

dcopy=data
#data=data.dropna()
#print(data)  


#TEMPERATURE
#group by zone id and temperature
dataf2=data.groupby(['propzoneid','temperature'],as_index=False, sort=False).first()
dataf2.shape
print(dataf2)
dataf2.columns


#finding average temperature of each zone
d1=dataf2.groupby(["propzoneid"],as_index=False)["temperature"].mean()
pd.DataFrame(d1).rename(columns={'temperature': 'avg zone tmp'}, inplace = True)
d1.shape
d1.columns
print(d1)

#finding total average of temperature of all zones
d1_avg=d1['avg zone tmp'].mean()
print(d1_avg)


d=dataf2.groupby(['datetime'],as_index=False, sort=False).first()
print(d)

#converting outdoor_temp column datatype from string to float
d['outdoor_temp'] = d['outdoor_temp'].replace('',0, regex=True)
X_np = d['outdoor_temp'].values
# set the object type as float
d['outdoor_temp'] = X_np.astype(float)

#declare a variable for different zones
zones=[2,14]

from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import r2_score

#TEMPERATURE prediction
i=0
print("Temperature Prediction for each zone")

for i in zones:
    print("--------------------")
    print('\n')
        
    print("zone:", i)
    ds=d.loc[(d["propzoneid"] == i),['datetime','temperature', 'humidity', 'occupancy',
       'outdoor_temp']]
    #print(ds)
    #To divide data into attributes and labels
    x1 = ds.drop(['temperature'],axis=1)
    y1 = ds['temperature']
    print('x1','\n')
    print(x1)
    print("--------------------")
    print('y1','\n')
    print(y1)
    
    train_x,test_x,train_y,test_y=train_test_split(x1,y1,test_size=0.05,train_size=0.95,shuffle=False)
    #print("test_x",test_x)
    #print("test_y",test_y)


    # DT REGRESSION
    # import the regressor
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state = False) 
      
    # fit the regressor with X and Y data
    regressor.fit(train_x, train_y)
    pred1=regressor.predict(test_x)
    print("--------------------")
    
    #df_results1=pd.DataFrame({"parameters":list(test_x)})
    #print(df_results1)           
    df_results2=pd.DataFrame({'datetime':list(test_x['datetime']),"actual":list(test_y),"predicted":list(pred1)})
    print(df_results2)

    s1=r2_score(test_y,pred1)
    print("--------------------")
    print('score=',s1)
