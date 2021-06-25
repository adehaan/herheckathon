#!/usr/bin/env python
# coding: utf-8

# In[413]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

import collections
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection
from sklearn.model_selection import train_test_split


# In[414]:


# STEP 1a Data Upload

# files last upladed in the folder
def upload_user():
    # upload last file
    # split file into train & test
    print('hi')


def upload():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data

def show_head(data):
    data.head(10)
    
    
train_data, test_data = upload()


# In[411]:


# STEP 1b Data Visualization

# 1 Identify type of variable
# histograms for all (max of 15)
# max/mean/min - boxplots - in one graph? or for every variable sperately



def histogramm(data,variable):
    sns.displot(data, x=variable)
    # TODO
    # save as image file
    # send to some public folder - folder "filename"

# different functions for different graphs
    

histogramm(train_data, 'education')
histogramm(train_data, "education")
histogramm(train_data, "occupation")
histogramm(train_data, "loan_size")


#plt.title("KMeans #cluster = 3")
#plt.xlabel('loan_size')
#plt.ylabel('sex')
#plt.scatter(train_data['loan_size'], train_data['education'], c=estimator.labels_)
#plt.show()



g = sns.FacetGrid(train_data, col='Continent', col_wrap=3, height=4)
g = (g.map(plt.hist, "Life Ladder", bins=np.arange(2,9,0.5)))


# In[415]:


# STEP 1c Preprocessing the data
# Normalize Dataset

#print(train_data2['occupation'].value_counts())

cleanup_nums = {"ZIP":     {"MT01RA": 0, "MT15PA": 1, "MT04PA":2, "MT12RA":3},
                "occupation": {"MZ10CD": 0, "MZ01CD": 1, "MZ11CD": 2}}

train_data2 = train_data.replace(cleanup_nums)
test_data2 = test_data.replace(cleanup_nums)

test_data2 = test_data2.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
train_data2 = train_data2.drop(['Unnamed: 0'], axis=1)

test_data2.head()


# In[421]:


test_data2


# In[416]:


# STEP 2a Splitting Datasets for Training

def upload_models():
    print('uploading models ...')

def split_data(data):
    data_wlabel = data.loc[:,data.columns != 'default']
    label = data['default']
    return data_wlabel, label
   


# In[418]:


# STEP 2b Models for Classification Tasks
# Naive Bayes, SVM, Logistic Regression, Neural Networks


# Naive Bayes
def nb_model(train_data, test_data):
    gnb = GaussianNB()
    X, label = split_data(train_data)
    Y, real = split_data(test_data)
    gaus_pred = gnb.fit(X, label)
    predict = gaus_pred.predict(Y)
    print(predict)
    wrong = (real != predict).sum()
    return real, predict


# Logistic Regression
def logistic_reg_model(train_data, test_data):
    logmodel = LogisticRegression()
    X, label = split_data(train_data)
    Y, real = split_data(test_data)
    log_pred = logmodel.fit(X,label)
    predict = log_pred.predict(Y)
    print(predict)
    wrong = (real != predict).sum()
    return real, predict


# Random Forest
def rf_model(train_data, test_data):
    X, label = split_data(train_data)
    Y, real = split_data(test_data)
    clf = RandomForestClassifier(max_depth=5, random_state=1)
    clf.fit(X, label)
    predict = clf.predict(Y)
    return real, predict

#Nearest Centroid
def nc_model(train_data, test_data):
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(weighting_data_train, weighting_target_train)
    predictions = nearest_centroid.predict(weighting_data_test)
    print("nearest_centroid: acc: {}".format(accuracy_score(weighting_target_test, predictions)))
    

# SVN

def evaluation_model(real, predict):
    acc = accuracy_score(real, predict)
    recall = recall_score(real, predict, average='weighted')
    precision = precision_score(real, predict, average='weighted')
    f1 = f1_score(real, predict, average='weighted')
    return acc, recall, precision, f1



# todo: how to choose which model?
model = 'randomf'
train_data = train_data2
test_data = test_data2

def run_model(model,train_data, test_data):
    
    #runs models & create predictions
    if model == 'nb_model':
        real, predict = rf_model(train_data,test_data)
    if model == 'logistic':
        real, predict = logistic_reg_model(train_data,test_data)
    if model == 'randomf':
        real, predict = rf_model(train_data,test_data)
    if model == 'svn':
        real, predict = rf_model(train_data,test_data)
    
    #evaluates model
    acc, recall, precision, f1 = evaluation_model(real, predict)
    
    return acc, recall, precision, f1, real, predict
        
        


acc, recall, precision, f1, real, predict = run_model(model, train_data2, test_data2)


#print(collections.Counter(predict))
#print(collections.Counter(real))


# In[419]:


# Calculating Differences in Prediction

def check_diff_sex():
    test_data2['predict']=pd.Series(predict)
    women_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["sex"]==1]
    men_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["sex"]==0]
    return women_f.shape[0], men_f.shape[0]


def check_diff_minority():
    test_data2['predict']=pd.Series(predict)
    min_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["minority"]==1]
    notmin_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["minority"]==0]
    return min_f.shape[0], notmin_f.shape[0]

women_f, men_f = check_diff_sex()
min_f, notmin_f = check_diff_minority()

print(women_f)
print(men_f)
print('----')
print(min_f)
print(notmin_f)


# In[409]:


data = [test_data2["default"], test_data2["predict"], test_data2['sex']]
headers = ["default", "predict", "sex"]
df3 = pd.concat(data, axis=1, keys=headers)

df3

df3 = df3.loc[(df3["default"] != df3["predict"])]
df3

print(collections.Counter(predict))


# In[433]:


# STEP 2 Train the models to 3 Subsets of the data


# 1 Female
def model_female(test_data2):
    train_data_female = train_data2.loc[train_data2["sex"]==1]
    test_data2.drop(['predict'], axis=1)
    f_acc, f_recall, f_precision, f_f1, f_real, f_predict = run_model(model,train_data_female, test_data2)
    return f_acc, f_recall, f_precision, f_f1, f_real, f_predict

# 2 Male
def model_male(test_data2):
    train_data_male = train_data2.loc[train_data2["sex"]==0]
    test_data2.drop(['predict'], axis=1)
    m_acc, m_recall, m_precision, m_f1, m_real, m_predict = run_model(model,train_data_male, test_data2)
    return m_acc, m_recall, m_precision, m_f1, m_real, m_predict


#def comparison_m_f_all():
m_acc, m_recall, m_precision, m_f1, m_real, m_predict = model_male(test_data2)
f_acc, f_recall, f_precision, f_f1, f_real, f_predict = model_female(test_data2)
    
print(f_accuracy)
print(m_accuracy)
print(accuracy)
print('------------')

print(f_f1)
print(m_f1)


# Idea: Mark these scores red, yellow or green based on certain threshold


# In[434]:


print(collections.Counter(predict))
print(collections.Counter(real))
print('---')
print(collections.Counter(f_predict))
print(collections.Counter(f_real))
print('---')
print(collections.Counter(m_predict))
print(collections.Counter(m_real))
print('---')


# In[206]:


# STEP 3 Optimization

# Step 3.1 Correlation Calculation


def correlation(data, feature):

  #corr_pearson = data['income'].corr(data['occupation'], method ='pearson')
    corr_pearson = data.corrwith(data[feature], method ='pearson')
    print('Pearson\'s correlation')
    print(corr_pearson)
    #print('spearman\'s correlation')
    #corr_spe = data.corrwith(data['sex'], method ='spearman')
    #print(corr_spe)
    aux_list = []
    for i in range(corr_pearson.shape[0]):
        if corr_pearson[i] >0.6 or corr_pearson[i] <-0.6:
            print(corr_pearson[i])
            print(data.columns[i])
            aux_list.append(data.columns[i])
    aux_list.remove(feature)
    return aux_list

feature = 'sex'
correlation(train_data2, feature)
    
        
# Step 3.2 Normalization when correlation above certain threshold

def normalization(data):
    # subset data - only normalize high correlation
    correlated_features = correlation(data, feature)
    # subset data - only normalize high correlation
    female_subset = data.loc[data[feature]==1]
    male_subset = data.loc[data[feature]==0]
    female_subset_normalized = female_subset.copy()
    male_subset_normalized = male_subset.copy()
    for feat in correlated_features:
        plot_histograms(data, feat, feature, 'Before')
        #female_subset_normalized[feat]= preprocessing.normalize(female_subset[feat], norm='l2')
        female_subset_normalized[feat] = female_subset_normalized[feat]/female_subset_normalized[feat].abs().max()
        
        male_subset_normalized[feat] = male_subset_normalized[feat]/male_subset_normalized[feat].abs().max()
        print(feat)
        data_normalized = female_subset_normalized.append(male_subset_normalized)
        plot_histograms(data_normalized, feat, feature, 'After')
    print(max(female_subset_normalized['education']))
    histogramm(female_subset_normalized, 'education')
normalization(train_data2, feature)



    
# Idea: Sampling - subset where we exclude outliers or replace by mean
# Idea: Creating artificial people
# Idea: change datapoints in training set
# Idea: feature generation/ selection
    
    
# Normalize all variables in a different way


# In[17]:


#estimator = KMeans(n_clusters = 3)
#labels = estimator.fit_predict(train_data[['loan_size', 'education']])
#print(labels)

# OR

#estimator.fit(train_data[['loan_size', 'education']])
#print(estimator.labels_)

