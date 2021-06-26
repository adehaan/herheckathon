import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import json

import collections
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn import preprocessing, svm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def upload():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data


def normalization(data):
    data_normalized = data.copy()
    for i in range(len(data.columns)):
        if data[data.columns[i]].dtype != 'int64' and data[data.columns[i]].dtype != 'bool':
            feat = data.columns[i]
            data_normalized[feat] = data_normalized[feat] / data_normalized[feat].abs().max()
    return data_normalized

def violin_plot(data, x, y, hue):
    palette_her = ["#FFC2C0", "#141B38", "#DE7D7E", "#626380", "#D0A0A5"]
    sns.set_palette(palette=palette_her)
    sns.set_style("darkgrid")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.set(font_scale=1)
    g = sns.catplot(x="sex", y="income", hue = 'default', split = True, kind="violin", inner=None, data=data, palette = palette_her, aspect = 0.8)
    g.set_xlabels(x, fontsize = 15)
    g.set_ylabels(y, fontsize = 15)
    sns.despine(offset=10, trim=True)
    g.savefig('violin_plot.png')
#violin_plot(aux, 'sex', 'income', 'default')

def plot_histograms(data1, feature, feat_label, info):
    palette_her = ["#FFC2C0", "#141B38", "#DE7D7E", "#626380", "#D0A0A5"]
    sns.set_style("darkgrid")
    sns.set_palette(palette=palette_her)
    sns.histplot(data1, x=feature, hue=feat_label, alpha = 1)
    sns.despine(offset=10, trim=True)
    plt.title(info+' Normalization', fontsize = 20)
    plt.savefig('figure_'+feature+'_'+ info+ '_normalization.png')


# MODELS

def split_data(data):
    data_wlabel = data.loc[:, data.columns != 'default']
    label = data['default']
    return data_wlabel, label


# Naive Bayes
def nb_model(train_data, test_data):
    gnb = GaussianNB()
    X, label = split_data(train_data)
    Y, real = split_data(test_data)
    gaus_pred = gnb.fit(X, label)
    predict = gaus_pred.predict(Y)
    wrong = (real != predict).sum()
    return real, predict


# Logistic Regression
def logistic_reg_model(train_data, test_data):
    logmodel = LogisticRegression()
    X, label = split_data(train_data)
    Y, real = split_data(test_data)
    log_pred = logmodel.fit(X, label)
    predict = log_pred.predict(Y)
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


# # Nearest Centroid
# def nc_model(train_data, test_data):
#     nearest_centroid = NearestCentroid()
#     nearest_centroid.fit(weighting_data_train, weighting_target_train)
#     predictions = nearest_centroid.predict(weighting_data_test)
#     print("nearest_centroid: acc: {}".format(accuracy_score(weighting_target_test, predictions)))


# SVN
def svm_model(train_data, test_data):
    X, label = split_data(train_data)
    Y, real = split_data(test_data)
    clf = svm.SVC()
    clf.fit(X, label)
    predict = clf.predict(Y)
    return real, predict


# kNN
def knn_model(train_data, test_data):
    X, label = split_data(train_data)
    Y, real = split_data(test_data)
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf.fit(X, label)
    predict = clf.predict(Y)
    return real, predict


# decision tree
def dt_model(train_data, test_data):
    print(train_data['default'].value_counts())
    X, label = split_data(train_data)
    print(collections.Counter(label))
    Y, real = split_data(test_data)
    clf = DecisionTreeClassifier()
    clf.fit(X, label)
    predict = clf.predict(Y)
    return real, predict


# evaluation function
def evaluation_model(real, predict):
    acc = accuracy_score(real, predict)
    recall = recall_score(real, predict, average='weighted')
    precision = precision_score(real, predict, average='weighted')
    f1 = f1_score(real, predict, average='weighted')
    return acc, recall, precision, f1


# -----------------------------------------------------

def run_model(model, train_data, test_data):
    # runs models & create predictions
    if model == 'nb_model':
        real, predict = nb_model(train_data, test_data)
    if model == 'logistic':
        real, predict = logistic_reg_model(train_data, test_data)
    if model == 'randomf':
        real, predict = rf_model(train_data, test_data)
    if model == 'svn':
        real, predict = svm_model(train_data, test_data)
    if model == 'knn':
        real, predict = knn_model(train_data, test_data)
    if model == 'dt':
        real, predict = dt_model(train_data, test_data)

    # evaluates model
    acc, recall, precision, f1 = evaluation_model(real, predict)

    return acc, recall, precision, f1, real, predict


# Calculating Differences in Prediction
def check_diff_sex(test_data2):
    # new method
    test_data2['real'] = real
    test_data2['predict'] = predict
    men = test_data2[test_data2['sex'] == 0]
    women = test_data2[test_data2['sex'] == 1]
    r_men = men['real']
    r_women = women['real']
    p_men = men['predict']
    p_women = women['predict']
    tn, fp, fn, tp = confusion_matrix(real, predict).ravel()
    print(fp / len(predict))
    tn1, fp1, fn1, tp1 = confusion_matrix(r_men, p_men).ravel()
    men_f = fp1 / len(p_men)
    print(fp1 / len(p_men))
    tn2, fp2, fn2, tp2 = confusion_matrix(r_women, p_women).ravel()
    women_f = fp2 / len(p_women)
    print(fp2 / len(p_women))

    test_data2 = test_data2.drop(['predict'], axis=1)
    test_data2 = test_data2.drop(['real'], axis=1)

    # old method
    # test_data2['predict']=pd.Series(predict)
    # women_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["sex"]==1]
    # men_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["sex"]==0]
    # test_data2 = test_data2.drop(['predict'], axis=1)
    return test_data2, women_f, men_f


def check_diff_minority(test_data2):
    # new method
    test_data2['real'] = real
    test_data2['predict'] = predict
    minority = test_data2[test_data2['minority'] == 1]
    notminority = test_data2[test_data2['minority'] == 0]
    r_min = minority['real']
    r_notmin = notminority['real']
    p_min = minority['predict']
    p_notmin = notminority['predict']

    tn3, fp3, fn3, tp3 = confusion_matrix(r_min, p_min).ravel()
    min_f = fp3 / len(p_min)

    tn4, fp4, fn4, tp4 = confusion_matrix(r_notmin, p_notmin).ravel()
    notmin_f = fp4 / len(p_notmin)

    test_data2 = test_data2.drop(['predict'], axis=1)
    test_data2 = test_data2.drop(['real'], axis=1)
    # old method
    # test_data2['predict']=pd.Series(predict)
    # min_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["minority"]==1]
    # notmin_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["minority"]==0]
    # test_data2 =test_data2.drop(['predict'], axis=1)
    return test_data2, min_f, notmin_f


# Rerun Models on Female subset or Male subset
# 1 Female
def model_female(test_data2, train_data2):
    train_data_female = train_data2.loc[train_data2["sex"] == 1]
    test_data_female = test_data2.loc[test_data2["sex"] == 1]
    # test_data2 = test_data2.drop(['predict'], axis=1, inplace=True)
    f_acc, f_recall, f_precision, f_f1, f_real, f_predict = run_model(model, train_data_female, test_data_female)
    return f_acc, f_recall, f_precision, f_f1, f_real, f_predict


# 2 Male
def model_male(test_data2, train_data2):
    train_data_male = train_data2.loc[train_data2["sex"] == 0]
    test_data_male = test_data2.loc[test_data2["sex"] == 0]
    # test_data2 = test_data2.drop(['predict'], axis=1)
    m_acc, m_recall, m_precision, m_f1, m_real, m_predict = run_model(model, train_data_male, test_data_male)
    return m_acc, m_recall, m_precision, m_f1, m_real, m_predict


# 3 Minority
def model_min(test_data2, train_data2):
    train_data_min = train_data2.loc[train_data2["minority"] == 1]
    test_data_min = test_data2.loc[test_data2["minority"] == 1]
    f_acc, f_recall, f_precision, f_f1, f_real, f_predict = run_model(model, train_data_min, test_data_min)
    return f_acc, f_recall, f_precision, f_f1, f_real, f_predict


# 4 Not Minority
def model_notmin(test_data2, train_data2):
    train_data_notmin = train_data2.loc[train_data2["minority"] == 0]
    test_data_notmin = test_data2.loc[test_data2["minority"] == 0]
    m_acc, m_recall, m_precision, m_f1, m_real, m_predict = run_model(model, train_data_notmin, test_data_notmin)
    return m_acc, m_recall, m_precision, m_f1, m_real, m_predict


# -------------------------------------------------------

# PREPROCESSING

# STEP 1 upload
train_data2, test_data2 = upload()
plot_histograms(train_data2, 'job_stability', 'default', 'info')
violin_plot(train_data2, 'sex', 'income', 'default')
# STEP 2 cleanup
enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(train_data2[['ZIP']]).toarray())  # merge with main df bridge_df on key values
enc_df = enc_df.rename(columns={0: 'Zip1', 1: 'Zip2', 2: 'Zip3', 3: 'Zip4'})
train_data2 = train_data2.join(enc_df)

enc_df2 = pd.DataFrame(
    enc.fit_transform(train_data2[['occupation']]).toarray())  # merge with main df bridge_df on key values
enc_df2 = enc_df2.rename(columns={0: 'Occupation1', 1: 'Occupation2', 2: 'Occupation3'})
train_data2 = train_data2.join(enc_df2)

enc_df3 = pd.DataFrame(enc.fit_transform(test_data2[['ZIP']]).toarray())  # merge with main df bridge_df on key values
enc_df3 = enc_df3.rename(columns={0: 'Zip1', 1: 'Zip2', 2: 'Zip3', 3: 'Zip4'})
test_data2 = test_data2.join(enc_df3)

enc_df4 = pd.DataFrame(
    enc.fit_transform(test_data2[['occupation']]).toarray())  # merge with main df bridge_df on key values
enc_df4 = enc_df4.rename(columns={0: 'Occupation1', 1: 'Occupation2', 2: 'Occupation3'})
test_data2 = test_data2.join(enc_df4)

train_data2 = train_data2.drop(['occupation', 'ZIP'], axis=1)
test_data2 = test_data2.drop(['occupation', 'ZIP'], axis=1)

test_data2 = test_data2.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
train_data2 = train_data2.drop(['Unnamed: 0'], axis=1)

# STEP 3 Normalization
# train_data2 = normalization(train_data2)
# test_data2 = normalization(test_data2)

# STEP 4: Train Model
# AT THE MOMENT: hardcode which model to use
# FUTURE: Include Button in Front-End to choose Model or run all and compare
model = 'randomf'

# STEP 5 Run and Evaluate Model
acc, recall, precision, f1, real, predict = run_model(model, train_data2, test_data2)

# STEP 6 Check for Biases
test_data2, women_f, men_f = check_diff_sex(test_data2)
test_data2, min_f, notmin_f = check_diff_minority(test_data2)

# STEP 7 Rerun Models on female and male only
m_acc, m_recall, m_precision, m_f1, m_real, m_predict = model_male(test_data2, train_data2)
f_acc, f_recall, f_precision, f_f1, f_real, f_predict = model_female(test_data2, train_data2)
min_acc, min_recall, min_precision, min_f1, min_real, min_predict = model_min(test_data2, train_data2)
notmin_acc, notmin_recall, notmin_precision, notmin_f1, notmin_real, notmin_predict = model_notmin(test_data2,
                                                                                                   train_data2)

# CREATES JSON FILE FOR FRONT-END
info = {'General_model': {'name': 'All', 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1},
        'Male_model': {'name': 'male', 'm_acc': m_acc, 'm_recall': m_recall, 'm_precision': m_precision, 'm_f1': m_f1},
        'Female_model': {'name': 'Female', 'f_acc': f_acc, 'f_recall': f_recall, 'f_precision': f_precision,
                         'f_f1': f_f1},
        'women_f': women_f, 'men_f': men_f, 'min_f': min_f, 'notmin_f': notmin_f}

with open('info.json', 'w') as outfile:
    json.dump(info, outfile)