import sys
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# loading dataset using pandas
games=pandas.read_csv(r'C:\Users\pc\Desktop\Machine learning pros\Projects on ML\Board game review predictor\games.csv')

# explore the dataset
print(games.columns)
print(games.shape)


print(games[games['average_rating']==0].iloc[0])
print(games[games['average_rating'] > 0].iloc[0])

games=games[games['users_rated']>0]
games=games.dropna(axis=0)
plt.hist(games['average_rating'])
plt.show()

corrmat=games.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

columns=games.colums.tolist()

columns=[c for c in columns if c not in ['bayes_average_rating','average_rating','type','name','id']]
target="average_rating"
train=games.sample(frac=0.8,random_state=1)

#select not in training to test

test=games.loc[~games.index.isin(train.index)]

print(train.shape)
print(test.shape)

#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LR=LinearRegression()

LR.fit(train[columns],train[target])

#predictions for test set by LR

prediction=LR.predict(test[columns])

mean_squared_error(prediction,test[target])

#Random forest regressor (collection of decision trees)
from sklearn.ensemble import RandomForestRegressor

RF=RandomForestRegressor(n_estimators=100, min_samples_leaf=10,random_state=1)

RF.fit(train[columns],train[target])
# for non linear models
# predictions for test ste by RFR
prediction2=RF.predict(test[columns])

mean_squared_error(prediction2,test[target])

rating= LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating2= RFR.predict(test[columns].iloc[0].values.reshape(1,-1))