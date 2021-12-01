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
import os

FILE='results.txt'

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

def test_regression(model, dt_test, X_test, postprocess=False):
    prediction = model.predict(X_test)
    if postprocess:
        prediction = postprocessing(dt_test, prediction)

    print(dt_test['gs'].shape)
    print(prediction.shape)
    return pearsonr(dt_test['gs'], prediction)[0]

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

def main_experiment(lexical=False, syntactic=False, postprocess=False, distance='jaccard'):
    file_object = open(FILE, 'w')

    experiment = f'\nExperiment with lexical={lexical}, syntactic={syntactic} and postprocess={postprocess}, distance={distance}.'
    print(experiment)
    file_object.write(experiment + '\n')
    dt_train, dt_test = readData.read_data()
    metrics_train = m.get_metrics(dt_train, lexical=lexical, syntactic=syntactic, distance=distance)
    metrics_test = m.get_metrics(dt_test, lexical=lexical, syntactic=syntactic, distance=distance)

    sort_metrics(dt_train, metrics_train)
    metrics_train = preprocess_metrics(metrics_train)
    X = DataFrame.from_dict(metrics_train)
    Y = dt_train['gs']
    metrics_test = preprocess_metrics(metrics_test)
    X_test = DataFrame.from_dict(metrics_test)

    results = []
    chosen_metrics = []

    regr = linear_model.LinearRegression()
    total = len(metrics_train.keys())
    for i in range(1, total):
        sfs = SequentialFeatureSelector(regr, n_features_to_select=i, n_jobs=multiprocessing.cpu_count())
        sfs.fit(X, Y)
    
        X_train = sfs.transform(X)
        X_test_final = sfs.transform(X_test)

        model = training_regression(dt_train, X_train)
        p_correlation = test_regression(model, dt_test, X_test_final, postprocess=postprocess)
        results.append(p_correlation)
        chosen_metrics.append(X_train.keys())

        result = f'\tChosen metrics: {X_train.shape[1]}, Pearson correlation: {p_correlation}'
        print(result)
        file_object.write(result + '\n')

    # All metrics
    model = training_regression(dt_train, X)
    p_correlation = test_regression(model, dt_test, X_test, postprocess=postprocess)
    results.append(p_correlation)
    chosen_metrics.append(X.keys())
    result = f'\tChosen metrics: {X.shape[1]}, Pearson correlation: {p_correlation}'
    print(result)
    file_object.write(result + '\n')

    max_index = results.index(max(results))
    final_result = f'\nBEST SETUP: {max_index+1} metrics, Correlation={results[max_index]}'
    metrics = f'{chosen_metrics[max_index]}'
    print(final_result)
    print(metrics)
    file_object.write(final_result + '\n')
    file_object.write(metrics + '\n')
    
    file_object.close()

if __name__ == '__main__':
    if os.path.exists(FILE):
        os.remove(FILE)
    
    #final_experiment()
    main_experiment(lexical=False, syntactic=False, postprocess=False, distance='jaccard')
    #main_experiment(lexical=False, syntactic=False, postprocess=False, distance='jaccard')
    #main_experiment(lexical=False, syntactic=False, postprocess=False, distance='jaccard')
    #main_experiment(lexical=False, syntactic=False, postprocess=False, distance='jaccard')
