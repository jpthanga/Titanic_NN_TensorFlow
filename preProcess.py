import numpy as np
from pandas import read_csv as read_csv

tr = read_csv("train.csv")

columns = list(tr)

drop = []
#Name
if 'Name' in columns:
    drop.append('Name')

#Ticket
if 'Ticket' in columns:
    drop.append('Ticket')

#Sex
if 'Sex' in columns:
    map = {'male' : 1,'female' : 0}
    tr['Male'] = tr['Sex'].map(map)

    map = {'male' : 0,'female' : 1}
    tr['Female'] = tr['Sex'].map(map)

    drop.append('Sex')

#Age - has nulls
ind = tr['Age'].isna()
rands = np.random.rand(ind.sum())*100
tr.loc[ind,'Age'] = rands

#Cabins
#Sex
if 'Cabin' in columns:
    tr['Cabin'] = tr['Cabin'].fillna(0)
    map = {0 : 0}
    tr['Cab'] = tr['Cabin'].map(map)
    tr['Cab'] = tr['Cab'].fillna(1)
    drop.append('Cabin')

#Embarked
drop.append('Embarked')
tr = tr.drop(drop,axis=1)


map = {1 : 0,0 : 1}
tr['Died'] = tr['Survived'].map(map)


Id = tr.as_matrix(['PassengerId'])
label = tr.as_matrix(['Survived','Died'])
data = tr.as_matrix(['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Male', 'Female', 'Cab'])
#
# print(Id)
print(label)
# print(data)

np.save("id",Id)
np.save("labels",label)
np.save("data",data)