import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

from sklearn.naive_bayes import GaussianNB
import collections
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn import preprocessing, svm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder


def upload():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data


def split_data(data):
    data_wlabel = data.loc[:, data.columns != 'default']
    label = data['default']
    return data_wlabel, label


def plot_histograms(data1, feature, feat_label, info):
    sns.displot(data1, x=feature, hue=feat_label)
    plt.title(info + ' normalization')
    plt.show()


# Correlation Calculation
def correlation(data, feature):
    # corr_pearson = data['income'].corr(data['occupation'], method ='pearson')
    corr_pearson = data.corrwith(data[feature], method='pearson')
    print('Pearson\'s correlation')
    print(corr_pearson)
    # print('spearman\'s correlation')
    # corr_spe = data.corrwith(data['sex'], method ='spearman')
    # print(corr_spe)
    aux_list = []
    for i in range(corr_pearson.shape[0]):
        if corr_pearson[i] > 0.6 or corr_pearson[i] < -0.6:
            print(corr_pearson[i])
            print(data.columns[i])
            aux_list.append(data.columns[i])
    aux_list.remove(feature)
    return aux_list


# Custom normalization when correlation above certain threshold
def custom_normalization(data_train, data_test, feature):
    # Correlation
    correlated_features = correlation(data_train, feature)
    # subset data - only normalize high correlation
    female_subset = data_train.loc[data_train[feature] == 1]
    male_subset = data_train.loc[data_train[feature] == 0]
    female_subset_train_normalized = female_subset.copy()
    male_subset_train_normalized = male_subset.copy()
    data_train_normalized = data_train.copy()

    female_subset = data_test.loc[data_train[feature] == 1]
    male_subset = data_test.loc[data_train[feature] == 0]
    female_subset_test_normalized = female_subset.copy()
    male_subset_test_normalized = male_subset.copy()
    data_test_normalized = data_test.copy()
    for i in range(len(data_train.columns)):
        if data_train.columns[i] in correlated_features and data_train[data_train.columns[i]].dtype != 'int64' and \
                data_train[
                    data_train.columns[i]].dtype != 'bool':
            feat = data_train.columns[i]
            plot_histograms(data_train, feat, feature, 'Before')
            female_subset_train_normalized[feat] = female_subset_train_normalized[feat] / \
                                                   female_subset_train_normalized[feat].abs().max()
            male_subset_train_normalized[feat] = male_subset_train_normalized[feat] / male_subset_train_normalized[
                feat].abs().max()
            print('custom_normalization', feat)
            data_train_normalized = female_subset_train_normalized.append(male_subset_train_normalized)
            plot_histograms(data_train_normalized, feat, feature, 'After')

            female_subset_test_normalized[feat] = female_subset_test_normalized[feat] / female_subset_test_normalized[
                feat].abs().max()
            male_subset_test_normalized[feat] = male_subset_test_normalized[feat] / male_subset_test_normalized[
                feat].abs().max()
            data_test_normalized = female_subset_test_normalized.append(male_subset_test_normalized)

    # for i in range(len(data_train.columns)):

    #   if data_train.columns[i] not in correlated_features and data_train[data_train.columns[i]].dtype != 'int64' and \
    #            data_train[data_train.columns[i]].dtype != 'bool':
    #        feat = data_train.columns[i]
    #       data_train_normalized[feat] = data_train_normalized[feat] / data_train_normalized[feat].abs().max()
    #       print('normalize', feat)

    #       data_test_normalized[feat] = data_test_normalized[feat] / data_test_normalized[feat].abs().max()
    return data_train_normalized, data_test_normalized


# MODELS

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


# evaluation function
def evaluation_model(real, predict):
    acc = accuracy_score(real, predict)
    recall = recall_score(real, predict, average='weighted')
    precision = precision_score(real, predict, average='weighted')
    f1 = f1_score(real, predict, average='weighted')
    return acc, recall, precision, f1


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
    train_data_female = train_data2.loc[train_data2["minority"] == 1]
    test_data_female = test_data2.loc[test_data2["minority"] == 1]
    f_acc, f_recall, f_precision, f_f1, f_real, f_predict = run_model(model, train_data_female, test_data_female)
    return f_acc, f_recall, f_precision, f_f1, f_real, f_predict


# 4 Not Minority
def model_notmin(test_data2, train_data2):
    train_data_male = train_data2.loc[train_data2["minority"] == 0]
    test_data_male = test_data2.loc[test_data2["minority"] == 0]
    m_acc, m_recall, m_precision, m_f1, m_real, m_predict = run_model(model, train_data_male, test_data_male)
    return m_acc, m_recall, m_precision, m_f1, m_real, m_predict


# -------------------------------------------------------

# PREPROCESSING

# STEP 1 upload
train_data2, test_data2 = upload()

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
feature = 'sex'
normalized_train, normalized_test = custom_normalization(train_data2, test_data2, feature)

# STEP 4: Train Model
# AT THE MOMENT: hardcode which model to use
# FUTURE: Include Button in Front-End to choose Model or run all and compare
model = 'randomf'

# STEP 5 Run and Evaluate Model
acc, recall, precision, f1, real, predict = run_model(model, normalized_train, normalized_test)

# STEP 6 Check for Biases
normalized_test, women_f, men_f = check_diff_sex(normalized_test)
normalized_test, min_f, notmin_f = check_diff_minority(normalized_test)

# STEP 7 Rerun Models on female and male only
m_acc, m_recall, m_precision, m_f1, m_real, m_predict = model_male(normalized_test, normalized_train)
f_acc, f_recall, f_precision, f_f1, f_real, f_predict = model_female(normalized_test, normalized_train)
min_acc, min_recall, min_precision, min_f1, min_real, min_predict = model_min(normalized_test, normalized_train)
notmin_acc, notmin_recall, notmin_precision, notmin_f1, notmin_real, notmin_predict = model_notmin(normalized_test,
                                                                                                   normalized_train)

# CREATE JSON FILE FOR THESE NUMBERS
info = {'General_model': {'name': 'All', 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1},
        'Male_model': {'name': 'male', 'm_acc': m_acc, 'm_recall': m_recall, 'm_precision': m_precision, 'm_f1': m_f1},
        'Female_model': {'name': 'Female', 'f_acc': f_acc, 'f_recall': f_recall, 'f_precision': f_precision,
                         'f_f1': f_f1},
        'women_f': women_f, 'men_f': men_f, 'min_f': min_f, 'notmin_f': notmin_f}

with open('info.json', 'w') as outfile:
    json.dump(info, outfile)