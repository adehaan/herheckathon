import pandas as pd

from sklearn.naive_bayes import GaussianNB
import collections
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def upload():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data


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

# Normalization when correlation above certain threshold
def custom_normalization(data, feature):
    correlated_features = correlation(data, feature)
    # subset data - only normalize high correlation
    female_subset = data.loc[data[feature] == 1]
    male_subset = data.loc[data[feature] == 0]
    female_subset_normalized = female_subset.copy()
    male_subset_normalized = male_subset.copy()
    for i in range(len(data.columns)):
        if data.columns[i] in correlated_features and data[data.columns[i]].dtype != 'int64' and data[
            data.columns[i]].dtype != 'bool':
            feat = data.columns[i]
            plot_histograms(data, feat, feature, 'Before')
            # female_subset_normalized[feat]= preprocessing.normalize(female_subset[feat], norm='l2')
            female_subset_normalized[feat] = female_subset_normalized[feat] / female_subset_normalized[feat].abs().max()
            male_subset_normalized[feat] = male_subset_normalized[feat] / male_subset_normalized[feat].abs().max()
            print('custom_normalization', feat)
            data_normalized = female_subset_normalized.append(male_subset_normalized)
            plot_histograms(data_normalized, feat, feature, 'After')

    for i in range(len(data.columns)):
        if data.columns[i] not in correlated_features and data[data.columns[i]].dtype != 'int64' and data[
            data.columns[i]].dtype != 'bool':
            feat = data.columns[i]
            data_normalized[feat] = data_normalized[feat] / data_normalized[feat].abs().max()
            print('normalize', feat)
    # print('max',max(female_subset_normalized['education']))
    return data_normalized


#MODELS

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

# Nearest Centroid
def nc_model(train_data, test_data):
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(weighting_data_train, weighting_target_train)
    predictions = nearest_centroid.predict(weighting_data_test)
    print("nearest_centroid: acc: {}".format(accuracy_score(weighting_target_test, predictions)))


# SVN


#kNN



# evaluation function
def evaluation_model(real, predict):
    acc = accuracy_score(real, predict)
    recall = recall_score(real, predict, average='weighted')
    precision = precision_score(real, predict, average='weighted')
    f1 = f1_score(real, predict, average='weighted')
    return acc, recall, precision, f1

#-----------------------------------------------------

def run_model(model, train_data, test_data):
    # runs models & create predictions
    if model == 'nb_model':
        real, predict = rf_model(train_data, test_data)
    if model == 'logistic':
        real, predict = logistic_reg_model(train_data, test_data)
    if model == 'randomf':
        real, predict = rf_model(train_data, test_data)
    if model == 'svn':
        real, predict = rf_model(train_data, test_data)

    # evaluates model
    acc, recall, precision, f1 = evaluation_model(real, predict)

    return acc, recall, precision, f1, real, predict


# Calculating Differences in Prediction
def check_diff_sex(test_data2, predict):
    test_data2['predict']=pd.Series(predict)
    women_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["sex"]==1]
    men_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["sex"]==0]
    return women_f.shape[0], men_f.shape[0]

def check_diff_minority(test_data2, predict):
    test_data2['predict']=pd.Series(predict)
    min_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["minority"]==1]
    notmin_f = test_data2.loc[(test_data2["default"] != test_data2["predict"]) & test_data2["minority"]==0]
    return min_f.shape[0], notmin_f.shape[0]


# Rerun Models on Female subset or Male subset
# 1 Female
def model_female(test_data2, train_data2):
    train_data_female = train_data2.loc[train_data2["sex"]==1]
    test_data2.drop(['predict'], axis=1)
    f_acc, f_recall, f_precision, f_f1, f_real, f_predict = run_model(model,train_data_female, test_data2)
    return f_acc, f_recall, f_precision, f_f1, f_real, f_predict

# 2 Male
def model_male(test_data2, train_data2):
    train_data_male = train_data2.loc[train_data2["sex"]==0]
    test_data2.drop(['predict'], axis=1)
    m_acc, m_recall, m_precision, m_f1, m_real, m_predict = run_model(model,train_data_male, test_data2)
    return m_acc, m_recall, m_precision, m_f1, m_real, m_predict



#-------------------------------------------------------

#PREPROCESSING

# STEP 1 upload
train_data, test_data = upload()

# STEP 2 cleanup
cleanup_nums = {"ZIP":     {"MT01RA": 0, "MT15PA": 1, "MT04PA":2, "MT12RA":3},
                "occupation": {"MZ10CD": 0, "MZ01CD": 1, "MZ11CD": 2}}
train_data2 = train_data.replace(cleanup_nums)
test_data2 = test_data.replace(cleanup_nums)
test_data2 = test_data2.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
train_data2 = train_data2.drop(['Unnamed: 0'], axis=1)

# STEP 3 Normalization
feature = 'sex'
normalized_train = custom_normalization(train_data2, feature)
normalized_test = custom_normalization(test_data2, feature)

# STEP 4: Train Model
# AT THE MOMENT: hardcode which model to use
# FUTURE: Include Button in Front-End to choose Model or run all and compare
model = 'randomf'

# STEP 5 Run and Evaluate Model
acc, recall, precision, f1, real, predict = run_model(model, normalized_train, normalized_test)

# STEP 6 Check for Biases
women_f, men_f = check_diff_sex(normalized_test, predict)
min_f, notmin_f = check_diff_minority(normalized_test, predict)

# STEP 7 Rerun Models on female and male only
m_acc, m_recall, m_precision, m_f1, m_real, m_predict = model_male(normalized_test, normalized_train)
f_acc, f_recall, f_precision, f_f1, f_real, f_predict = model_female(normalized_test, normalized_train)

# TODO: CREATE JSON FILE FOR THESE NUMBERS