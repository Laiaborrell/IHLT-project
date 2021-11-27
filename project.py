from pandas.core.frame import DataFrame
import readData
from nltk.metrics import jaccard_distance
import metrics as m
from scipy.stats import pearsonr
import numpy as np
import operator
from sklearn import linear_model

def printTopMetrics(correlations):
    print('')
    for i, metric in enumerate(correlations):
        top=i+1
        correlation=round(correlations[metric], 2)
        print(f'Top {top}: {metric}, {correlation}')

def jacc_sim(list1, list2):
    return 1-jaccard_distance(set(list1), set(list2)) # 1-distance

def get_best_metrics(dt_train, metrics, n=5):
    metrics = preprocess_metrics(metrics)
    if n > len(metrics.values()):
        return metrics.keys()
    correlations = {}
    for metric in metrics:
        correlations[metric] = pearsonr(dt_train['gs'], metrics[metric])[0]
    sorted_c = dict(sorted(correlations.items(), key=operator.itemgetter(1), reverse=True))
    printTopMetrics(sorted_c)

    chosen_metrics = []
    for i in range(n):
        chosen_metrics.append(list(correlations.keys())[i])
    print(f'\nChosen metrics: {n}')

    return chosen_metrics

def preprocess_metrics(metrics):
    for key in metrics:
        metrics[key] = (metrics[key] - np.min(metrics[key])) / (np.max(metrics[key]) - np.min(metrics[key]))
    #    metrics[key] = np.log10(metrics[key])
    return metrics

def training_regression(dt_train, metrics):
    metrics = preprocess_metrics(metrics)
    X = DataFrame.from_dict(metrics)
    Y = dt_train['gs']
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    return regr

def test_regression(model, dt_test, metrics):
    metrics = preprocess_metrics(metrics)
    X = DataFrame.from_dict(metrics)
    prediction = model.predict(X)
    print('\nFINAL SCORE REGRESSION = {}'.format(pearsonr(dt_test['gs'], prediction)[0]))

if __name__ == '__main__':
    #READING THE DATA
    dt_train, dt_test = readData.read_data()

    metrics_train = m.get_metrics(dt_train)
    # aixo ens ho podriem estalviar heavy, nomes caldria computar les X metriques millors
    metrics_test = m.get_metrics(dt_test)

    chosen_metrics = get_best_metrics(dt_train, metrics_train, n=10)
    remove_metrics = []
    for key in metrics_train:
        if key not in chosen_metrics:
            remove_metrics.append(key)
    for rem_metric in remove_metrics:
        metrics_train.pop(rem_metric)
        metrics_test.pop(rem_metric)

    model = training_regression(dt_train, metrics_train)
    test_regression(model, dt_test, metrics_test)