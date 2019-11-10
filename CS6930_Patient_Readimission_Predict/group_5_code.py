"""
Created on Mon Dec  3 01:16:23 2018
@author: XYXS
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import sklearn
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
sklearn.exceptions.DataConversionWarning
from sklearn.linear_model import LogisticRegression

import copy 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
RANDOM_SEED = 42


def roc_auc_func(y_true, y_score):
    return roc_auc_score(y_true, y_score, average='weighted')
def print_res(fpr, tpr, thresholds, class_1_min_recall=.60):
    sens  = tpr[tpr>=class_1_min_recall]
    specs = 1 - fpr[tpr>=class_1_min_recall]
    thres = thresholds[tpr>=class_1_min_recall]

    print("Thresholds: ")
    print(thres[0])

    print("True positive rate:")
    print(sens[0])

    print("True negavetive rate:")
    print(specs[0])

#load data of filtered attibutes
data=pd.read_csv('data_preprocessing.csv')
x=data.iloc[:,[23, 13, 22, 51, 17, 20, 93, 92, 98, 99, 61, 62, 18, 95, 39, 29, 49,
       14, 21, 58, 96, 97, 43, 15,  8, 30, 11, 53,  9, 46, 59, 31, 19, 63,
       57, 54, 76, 25, 74, 45, 24, 56, 73, 86, 65, 67, 89, 85, 81, 90,  4,
       66, 52, 42, 32, 82, 34, 12, 44, 37, 40, 88, 16, 79, 38, 48, 80, 26,
        3, 84, 50, 71, 64,  2, 47, 36, 28, 33,  1,  5,  6, 72, 77, 27]]
y=data.iloc[:,100:101]
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(x, y, test_size=0.33, random_state=20)
ros = RandomOverSampler(random_state=0)

kfolds = 5
kf = KFold(n_splits=kfolds) 
kf.get_n_splits(X_TRAIN)

#find best parameters of different madels
#find best parameters of svm 
a=[0.01,0.1,1,2,3]
result=[]
for i in  a:
    final_auc=0
    for train_index, test_index in kf.split(X_TRAIN):
        train_x, test_x = X_TRAIN.iloc[train_index], X_TRAIN.iloc[test_index]
        train_y, test_y= Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[test_index]
        train_x, train_y = ros.fit_resample(train_x, train_y)
        clf1 = SVC(C=i, kernel='rbf')
        clf1.fit(train_x, train_y)
        y_pred = clf1.predict(test_x)
        temp=roc_auc_score(test_y,y_pred)
        final_auc=final_auc+temp
    print('The average roc_auc is',final_auc/5)
    result.append(final_auc/5)   
plt.plot(a,result)

kernel=['linear','poly','rbf']
result=[]
for i in kernel:
    final_auc=0
    for train_index, test_index in kf.split(X_TRAIN):
        train_x, test_x = X_TRAIN.iloc[train_index], X_TRAIN.iloc[test_index]
        train_y, test_y= Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[test_index]
        train_x, train_y = ros.fit_resample(train_x, train_y)
        clf1 = SVC(C=0.1, kernel=i)
        clf1.fit(train_x, train_y)
        y_pred = clf1.predict(test_x)
        temp=roc_auc_score(test_y,y_pred)
        final_auc=final_auc+temp
    print('The average roc_auc is',final_auc/5)
    result.append(final_auc/5)   
plt.plot(kernel,result)


#find best parameters of rf

a=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,None]
result=[]
for i in  a:
    final_auc=0
    for train_index, test_index in kf.split(X_TRAIN):
        train_x, test_x = X_TRAIN.iloc[train_index], X_TRAIN.iloc[test_index]
        train_y, test_y= Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[test_index]
        train_x, train_y = ros.fit_resample(train_x, train_y)
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=i,random_state=0)
        clf2.fit(train_x, train_y)
        y_pred = clf2.predict(test_x)
        fpr, tpr, thresholds = roc_curve(test_y,y_pred, pos_label=1)
        temp=roc_auc_score(test_y,y_pred)
        final_auc=final_auc+temp
    print('The average roc_auc is',final_auc/5)
    result.append(final_auc/5)   
plt.plot(a,result)


result=[]
b=['log2','sqrt','auto',1,3,5,7,9,11,13]
for i in  b:
    final_auc=0
    for train_index, test_index in kf.split(X_TRAIN):
        train_x, test_x = X_TRAIN.iloc[train_index], X_TRAIN.iloc[test_index]
        train_y, test_y= Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[test_index]
        train_x, train_y = ros.fit_resample(train_x, train_y)
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=7,max_features=i ,random_state=0)
        clf2.fit(train_x, train_y)
        y_pred = clf2.predict(test_x)
        fpr, tpr, thresholds = roc_curve(test_y,y_pred, pos_label=1)
        temp=roc_auc_score(test_y,y_pred)
        final_auc=final_auc+temp
    print('The average roc_auc is',final_auc/5)
    result.append(final_auc/5)   
plt.plot(b,result)

result=[]
c1=[10,100,1000]
c2=[70,75,80,85,90,95,100,105,110,115,120]
for i in  c1:
    final_auc=0
    for train_index, test_index in kf.split(X_TRAIN):
        train_x, test_x = X_TRAIN.iloc[train_index], X_TRAIN.iloc[test_index]
        train_y, test_y= Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[test_index]
        train_x, train_y = ros.fit_resample(train_x, train_y)
        clf2 = RandomForestClassifier(n_estimators=i, max_depth=7,max_features=11 ,random_state=0)
        clf2.fit(train_x, train_y)
        y_pred = clf2.predict(test_x)
        fpr, tpr, thresholds = roc_curve(test_y,y_pred, pos_label=1)
        temp=roc_auc_score(test_y,y_pred)
        final_auc=final_auc+temp
    print('The average roc_auc is',final_auc/5)
    result.append(final_auc/5)   
plt.plot(c1,result)
plt.plot(c2,result)


#find best parameters of knn
k_list= [2*n+1 for n in range(1,50)]
k_res=[]
for k in k_list:
    final_auc_knn=0
    knn = KNeighborsClassifier(n_neighbors=k)
    for train_index, test_index in kf.split(X_TRAIN):
        train_x, test_x = X_TRAIN[train_index], X_TRAIN[test_index]
        train_y, test_y= Y_TRAIN[train_index], Y_TRAIN[test_index]
        train_x, train_y = ros.fit_resample(train_x, train_y)
        
        knn.fit(train_x, train_y)
        y_pred = knn.predict(test_x)
        temp=roc_auc_score(test_y,y_pred)
        final_auc_knn = final_auc_knn+temp
    print("---------------------")
    #print("Result of knn ")
    print('When K=',k,'The average roc_auc is',final_auc_knn/5)
    k_res.append(final_auc_knn/5)
    y_pred_f = knn.predict(X_TEST)
    print(classification_report(Y_TEST,y_pred_f))




#find the best parameters of lr
fff=[]
final_auc=0
clist=[0.01,0.1,1,10,100]
for C1 in clist:
    final_auc=0
    for train_index, test_index in kf.split(X_TRAIN):
    #print(train_index, test_index)
        train_x, test_x = X_TRAIN[train_index], X_TRAIN[test_index]
        train_y, test_y= Y_TRAIN[train_index], Y_TRAIN[test_index]
        train_x, train_y = ros.fit_resample(train_x, train_y)
    
        clf = LogisticRegression(C=C1,penalty='l2')
        clf.fit(train_x, train_y)

        y_pred = clf.predict(test_x)
        temp=roc_auc_score(test_y,y_pred)
        final_auc=final_auc+temp
    
    print("---------------------")
    print('When C=',C1,'The average roc_auc is',final_auc/5)
    fff.append(final_auc/5)
    y_pred_f = clf.predict(X_TEST)
    print(classification_report(Y_TEST,y_pred_f))

strclist = ["0.01","0.1","1","10","100"]   
plt.plot(strclist,fff)


#To find best parameters to train models print the result with and without changing thresholds

#svm
svc = SVC(C=0.1, kernel='linear',probability=True)
X_TRAIN, Y_TRAIN=ros.fit_resample(X_TRAIN, Y_TRAIN)
svc.fit(X_TRAIN, Y_TRAIN)
svc_test_pred = svc.predict(X_TEST)[:,1]
print(classification_report(Y_TEST,svc_test_pred))
svc_pred_prob = svc.predict_proba(X_TEST)[:,1]
roc_auc=roc_auc_score(Y_TEST,svc_test_pred)
print('The test roc_auc is',roc_auc)
print(confusion_matrix(Y_TEST,svc_test_pred))
y_true = Y_TEST
y_probs = svc_pred_prob  
fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
for class_1_min_recall in [.70, .75, .80]:
    print("=============================")
    print("Sens >= %s :" % class_1_min_recall)
    print_res(fpr, tpr, thresholds, class_1_min_recall)
    print()    
change_threshold_svc=copy.copy(svc_pred_prob)   
for i in range(len(change_threshold_svc)):
    if change_threshold_svc[i]>=0.4:
        change_threshold_svc[i]=int(1)
    else:
        change_threshold_svc[i]=int(0)
print(classification_report(Y_TEST,change_threshold_svc))
roc_auc=roc_auc_score(Y_TEST,change_threshold_svc)
print('The test roc_auc is',roc_auc)
print(confusion_matrix(Y_TEST,change_threshold_svc))


#RandomForest
rf = RandomForestClassifier(n_estimators=110,max_features=11, criterion='gini', max_depth=7,random_state=0,n_jobs=-1)
X_TRAIN, Y_TRAIN=ros.fit_resample(X_TRAIN, Y_TRAIN)
rf.fit(X_TRAIN, Y_TRAIN)
rf_test_pred = rf.predict(X_TEST)
roc_auc=roc_auc_score(Y_TEST,rf_test_pred)
rf_pred_prob = rf.predict_proba(X_TEST)[:,1]
roc_auc=roc_auc_score(Y_TEST,rf_test_pred)
print(classification_report(Y_TEST,rf_test_pred))
print('The test roc_auc is',roc_auc)
print(confusion_matrix(Y_TEST,rf_test_pred))
y_true = Y_TEST
y_probs = rf_pred_prob
  
fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
for class_1_min_recall in [.70, .75, .80]:
    print("=============================")
    print("Sens >= %s :" % class_1_min_recall)
    print_res(fpr, tpr, thresholds, class_1_min_recall)
    print()    

change_threshold_rf=copy.copy(rf_pred_prob)
for i in range(len(change_threshold_rf)):
    if change_threshold_rf[i]>=0.475:
        change_threshold_rf[i]=int(1)
    else:
        change_threshold_rf[i]=int(0)
print(classification_report(Y_TEST,change_threshold_rf))
roc_auc=roc_auc_score(Y_TEST,change_threshold_rf)
print('The test roc_auc is',roc_auc)
print(confusion_matrix(Y_TEST,change_threshold_rf))



#Logistic Regression
lr = LogisticRegression(C=0.01,penalty='l2')
X_TRAIN, Y_TRAIN=ros.fit_resample(X_TRAIN, Y_TRAIN)
lr.fit(X_TRAIN, Y_TRAIN)
lr_test_pred=lr.predict(X_TEST)
lr_pred_prob = lr.predict_proba(X_TEST)[:,1]
roc_auc=roc_auc_score(Y_TEST,lr_test_pred)
print(classification_report(Y_TEST,lr_test_pred))
print('The test roc_auc is',roc_auc)
print(confusion_matrix(Y_TEST,lr_test_pred))
y_true = Y_TEST
y_probs = lr_pred_prob
fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
for class_1_min_recall in [.70, .75, .80]:
    print("=============================")
    print("Sens >= %s :" % class_1_min_recall)
    print_res(fpr, tpr, thresholds, class_1_min_recall)
    print()    

change_threshold_lr=copy.copy(lr_pred_prob)
for i in range(len(change_threshold_rf)):
    if change_threshold_lr[i]>=0.453245:
        change_threshold_lr[i]=int(1)
    else:
        change_threshold_lr[i]=int(0)
print(classification_report(Y_TEST,change_threshold_lr))
roc_auc=roc_auc_score(Y_TEST,change_threshold_lr)
print('The test roc_auc is',roc_auc)
print(confusion_matrix(Y_TEST,change_threshold_rf))



#KNN
knn = KNeighborsClassifier(n_neighbors=87)
knn.fit(X_TRAIN, Y_TRAIN)
knn_test_pred=knn.predict(X_TEST)
knn_pred_prob = knn.predict_proba(X_TEST)[:,1]
roc_auc=roc_auc_score(Y_TEST,knn_test_pred)
print(classification_report(Y_TEST,knn_test_pred))
print('The test roc_auc is',roc_auc)
print(confusion_matrix(Y_TEST,knn_test_pred))
y_true = Y_TEST
y_probs = knn_pred_prob
fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
for class_1_min_recall in [.70, .75, .80]:
    print("=============================")
    print("Sens >= %s :" % class_1_min_recall)
    print_res(fpr, tpr, thresholds, class_1_min_recall)
    print()    

change_threshold_knn=copy.copy(knn_pred_prob)
for i in range(len(change_threshold_knn)):
    if change_threshold_knn[i]>=0.47:
        change_threshold_knn[i]=int(1)
    else:
        change_threshold_knn[i]=int(0)
print(classification_report(Y_TEST,change_threshold_knn))
roc_auc=roc_auc_score(Y_TEST,change_threshold_knn)
print('The test roc_auc is',roc_auc)
print(confusion_matrix(Y_TEST,change_threshold_knn))




#Ensemble
#do ensemble before adjust the threshold of all algorithm

svc_test_pred=pd.read_csv('new.csv')
change_threshold_svc=pd.read_csv('new_0.7.csv')
svc_test_pred=np.array(svc_test_pred['0'],dtype=int)
change_threshold_svc=np.array(change_threshold_svc['0'],dtype=int)

y_pred=(change_threshold_svc+change_threshold_rf+change_threshold_lr)
for i in range(len(y_pred)):
    if y_pred[i]>=2:
        y_pred[i]=int(1)
    else:
        y_pred[i]=int(0)
roc_auc=roc_auc_score(Y_TEST,y_pred)
print('The test roc_auc is',roc_auc)
print(classification_report(Y_TEST,y_pred))
print(confusion_matrix(Y_TEST,y_pred))


#after change the threshold do the ensemble 
y_pred=(rf_test_pred+svc_test_pred+lr_test_pred)
for i in range(len(y_pred)):
    if y_pred[i]>=2:
        y_pred[i]=int(1)
    else:
        y_pred[i]=int(0)


roc_auc=roc_auc_score(Y_TEST,y_pred)
print('The test roc_auc is',roc_auc)
print(classification_report(Y_TEST,y_pred))
print(confusion_matrix(Y_TEST,y_pred))

