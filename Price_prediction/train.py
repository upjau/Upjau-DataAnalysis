import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle
import xgboost
import plotly.plotly as py 	
import plotly


#
df = pd.read_csv('./processed_data/data.csv')

y = np.array(df['AVG_Price'])
df=df.drop(['AVG_Price','Prevyear_open'],axis=1)







x=df.values

#splitting the data in train and test test
X_train,X_test,Y_train,Y_test =train_test_split(x,y,test_size=0.2)


###Linear Regression
lm = LinearRegression()

#fitting data in linear model
lm.fit(X_train,Y_train)


###xgboost regressor

test = df.values

xgb = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=5)

#fitting data in xgboost model
xgb.fit(X_train,Y_train)

#Predictinng on linear model
lm_pred=lm.predict(test)



xgb_pred = xgb.predict(test)



# printing accuracy of both models
accuracy = lm.score(test,y)
print("Regression_accuracy",accuracy)

accuracy = xgb.score(test,y)
print("xgb_accuracy",accuracy)



# making a list of columns in dataframe
columns=df.columns
columns = list(columns)
print(columns)

# out = open('columns.pickle','wb')
# pickle.dump(columns,out)

#print(predict)
data = {'lm_pred':lm_pred,'xgb':xgb_pred,'Price':y}
new_df = pd.DataFrame(data=data)

#printing actual and predicted price from both models
print(new_df)

##saving the trained model



# out = open('xgboostmodel98.pickle','wb')
# pickle.dump(xgb,out)






