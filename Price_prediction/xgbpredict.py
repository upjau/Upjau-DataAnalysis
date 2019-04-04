import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle
import xgboost




df = pd.read_csv('../processed_data/All_data.csv')

cor=df

a=df['Market Year']

b=df[['Market Year','AVG_Price',]]

y = np.array(df['AVG_Price'])

df=df.drop(['Market Year','AVG_Price','Prevyear_open'],axis=1)







x=df.values

#splitting the data into training and test sets
X_train,X_test,Y_train,Y_test =train_test_split(x,y,test_size=0.2)


###Linear Regression
lm = LinearRegression()

lm.fit(X_train,Y_train)

test = df.values

lm_pred=lm.predict(test)


# loading the saved xgboost trained model
xgb = pickle.load(open('xgbmodel98.pickle','rb'))

print(test)

xgb_pred = xgb.predict(test)




accuracy = lm.score(test,y)
print("Regression_accuracy",accuracy)

accuracy = xgb.score(test,y)
print("xgb_accuracy",accuracy)




columns=df.columns
columns = list(columns)
print(columns)

# out = open('columns.pickle','wb')
# pickle.dump(columns,out)

#creating a dataframe with predicted and actual values
data = {'Market_year':a,'lm_pred':lm_pred,'xgb':xgb_pred,'Price':y}
new_df = pd.DataFrame(data=data)
print(new_df)



# Plotting the actual and the predicted values
sns.set_style('whitegrid')
fig = plt.figure(figsize=(10,10))
ax = sns.lineplot(x="Market_year",y="lm_pred",label="lm",lw=2,data=new_df)
ax4= sns.lineplot(x="Market_year",y="xgb",color="coral",label="xgb",lw=2,data=new_df)
ax2=sns.lineplot(x="Market Year",y="AVG_Price",color="blue",label="Price",lw=2,data=b)

plt.legend()
plt.xticks(rotation=60)
plt.show()


corr = cor.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
