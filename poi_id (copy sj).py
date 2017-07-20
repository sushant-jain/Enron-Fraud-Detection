#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary','total_payments','from_this_person_to_poi','from_poi_to_this_person'] # You will need to use more features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
  'long_term_incentive', 'restricted_stock', 'director_fees',
	'to_messages',  'from_messages','from_poi_to_this_person', 'from_this_person_to_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL",0)
# data_dict.pop("LAY KENNETH L",0)

import matplotlib.pyplot

data = featureFormat(data_dict, features_list)



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#print features[0]

# xxx=np.delete(features,17,1)
# print xxx[0]
# from sklearn.preprocessing import MinMaxScaler
# mms=MinMaxScaler()
# features= mms.fit_transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
from sklearn.decomposition import PCA
pca=PCA(n_components=2)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


financial_features_train=[]
for i in features_train:
	financial_features_train.append(i[:-4])

financial_features_test=[]
for i in features_test:
	financial_features_test.append(i[:-4])

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
financial_features_train= mms.fit_transform(financial_features_train)
financial_features_test=mms.transform(financial_features_test)

financial_features_train = pca.fit_transform(financial_features_train)
financial_features_test=pca.transform(financial_features_test)

print pca.explained_variance_ratio_

email_features_train=[]
for i in features_train:
	if(i[-3]+i[-4]!=0.0):
		email_features_train.append(np.array([float(i[-2]/float(i[-3]+i[-4])),float(i[-1]/float(i[-3]+i[-4]))]))
	else:
		email_features_train.append(np.array([0,0]))

email_features_test=[]
for i in features_test:
	if(i[-3]+i[-4]!=0.0):
		email_features_test.append(np.array([float(i[-2]/float(i[-3]+i[-4])),float(i[-1]/float(i[-3]+i[-4]))]))
	else:
		email_features_test.append(np.array([0,0]))


# print zip(email_features_train,financial_features_train)
final_features_train=[]
for i,j in zip(financial_features_train,email_features_train):
	final_features_train.append([i[0],i[1],j[0],j[1]])
# print final_features_train

final_features_test=[]
for i,j in zip(financial_features_test,email_features_test):
	final_features_test.append([i[0],i[1],j[0],j[1]])


for point,ans in zip(final_features_train,labels_train):
    pc1 = point[0]
    pc2 = point[3]
    if(int(ans)==0):
    	matplotlib.pyplot.scatter( pc1,pc2,color="blue" )
    if(int(ans)==1):
    	matplotlib.pyplot.scatter(pc1,pc2,color="red" )
matplotlib.pyplot.xlabel("pc1")
matplotlib.pyplot.ylabel("ratio")
matplotlib.pyplot.show()

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(final_features_train,labels_train)
pred=clf.predict(final_features_test)

from sklearn.metrics import precision_score
print precision_score(labels_test,pred)

from sklearn.metrics import recall_score
print recall_score(labels_test,pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)