from pandas.core.frame import DataFrame
import readData
from nltk.metrics import jaccard_distance
import metrics as m
from scipy.stats import pearsonr
import numpy as np
import operator
from sklearn import linear_model
from scipy import stats
from sklearn.feature_selection import SequentialFeatureSelector
import multiprocessing

def printTopMetrics(correlations):
    print('')
    for i, metric in enumerate(correlations):
        top=i+1
        correlation=round(correlations[metric], 2)
        print(f'Top {top}: {metric}, {correlation}')

def jacc_sim(list1, list2):
    return 1-jaccard_distance(set(list1), set(list2)) # 1-distance

def sort_metrics(dt_train, metrics):
    metrics = preprocess_metrics(metrics)

    correlations = {}
    for metric in metrics:
        correlations[metric] = pearsonr(dt_train['gs'], metrics[metric])[0]
    sorted_c = dict(sorted(correlations.items(), key=operator.itemgetter(1), reverse=True))
    printTopMetrics(sorted_c)

    return sorted_c

def preprocess_metrics(metrics):
    for key in metrics:
        #metrics[key] = stats.zscore(metrics[key])
        metrics[key] = (metrics[key] - np.min(metrics[key])) / (np.max(metrics[key]) - np.min(metrics[key]))
        #metrics[key] = np.log10(metrics[key])
    return metrics

def postprocessing(dt_test, prediction):
    r, _ = dt_test.shape
    for i in range(r):
        a = ''.join(ch for ch in dt_test[0][i] if ch.isalnum())
        b = ''.join(ch for ch in dt_test[1][i] if ch.isalnum())    
        if a.lower() == b.lower():
            prediction[i] = np.max(prediction)
    return prediction
            
def training_regression(dt_train, X):
    Y = dt_train['gs']
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    return regr

def test_regression(model, dt_test, X_test):
    prediction = model.predict(X_test)
    #prediction = postprocessing(dt_test, prediction)
    print('FINAL SCORE REGRESSION = {}'.format(pearsonr(dt_test['gs'], prediction)[0]))

def final_experiment():
    #READING THE DATA
    dt_train, dt_test = readData.read_data()

    metrics_train = m.get_metrics(dt_train)
    metrics_test = m.get_metrics(dt_test)

    metrics_train = preprocess_metrics(metrics_train)
    X = DataFrame.from_dict(metrics_train)
    Y = dt_train['gs']
    metrics_test = preprocess_metrics(metrics_test)
    X_test = DataFrame.from_dict(metrics_test)

    regr = linear_model.LinearRegression()

    i = 25
    sfs = SequentialFeatureSelector(regr, n_features_to_select=i, n_jobs=multiprocessing.cpu_count())
    sfs.fit(X, Y)

    X_train = sfs.transform(X)
    X_test_final = sfs.transform(X_test)

    print(f'\nChosen metrics: {X_train.shape[1]}')

    model = training_regression(dt_train, X_train)
    test_regression(model, dt_test, X_test_final)

def main_experiment():
    #READING THE DATA
    dt_train, dt_test = readData.read_data()

    metrics_train = m.get_metrics(dt_train)
    metrics_test = m.get_metrics(dt_test)

    metrics_train = preprocess_metrics(metrics_train)
    X = DataFrame.from_dict(metrics_train)
    Y = dt_train['gs']
    metrics_test = preprocess_metrics(metrics_test)
    X_test = DataFrame.from_dict(metrics_test)

    regr = linear_model.LinearRegression()

    total = len(metrics_train.keys())
    for i in range(20, total):
        sfs = SequentialFeatureSelector(regr, n_features_to_select=i, n_jobs=multiprocessing.cpu_count())
        sfs.fit(X, Y)
    
        X_train = sfs.transform(X)
        X_test_final = sfs.transform(X_test)

        print(f'\nChosen metrics: {X_train.shape[1]}')
        model = training_regression(dt_train, X_train)
        test_regression(model, dt_test, X_test_final)

    # All metrics
    print(f'\nChosen metrics: {X.shape[1]}')
    model = training_regression(dt_train, X)
    test_regression(model, dt_test, X_test)
    


if __name__ == '__main__':
    #final_experiment()
    main_experiment()
