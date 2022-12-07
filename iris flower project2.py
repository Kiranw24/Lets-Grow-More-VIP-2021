import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

features = ['SL','SW','PL','PW']
df = pd.read11-csv('',names = features)
#print(df.shape)
#print(df.head(10))
#print(df.describe())
df.plot(kind = 'box',subplots= True,layout = (2,2))
plt.show()

from sklearn.preprocessing import LabelEncoder
df['class'] = LabelEncoder().fit_transform(df['class'])

import seaborn as sns
sns.heatmap(df.corr(),annot=True)
plt.show()

data = df.values
ip = data[:,0:4]
op = data[:,4]

from sklearn.model_selection import train_test_split
ti,vi,to,vo = train_test_split(ip,op,test_size = 0.2,random_state=1)

#print(ti)
#print(vi)
#print(vo)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(solver = 'liblinear',multi_class='ovr')

#support vector machine
from sklearn.svm import SVC
#model = SVC(gamma='auto')

#Linear Discreminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#model = LinearDiscriminantAnalysis()

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(ti,to)
predictions = model.predict(vi)

from sklearn.metrics import accuracy_score
print(accuracy_score(vo,predictions))

#Conusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(vo,predictions)

from sklearn.metrics import classification_report
print(classification_report(vo,predictions))
