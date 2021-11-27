from pandas.core.frame import DataFrame
import readData
from nltk.metrics import jaccard_distance
import metrics as m
from scipy.stats import pearsonr
import numpy as np
import operator
from sklearn import linear_model
from scipy import stats

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
    print('FINAL SCORE REGRESSION = {}'.format(pearsonr(dt_test['gs'], prediction)[0]))

def final_experiment():
    #READING THE DATA
    dt_train, dt_test = readData.read_data()

    metrics_train = m.get_metrics(dt_train)
    metrics_test = m.get_metrics(dt_test)
    sorted_metrics = sort_metrics(dt_train, metrics_train)

    n = 10
    final_metrics_train = {}
    final_metrics_test = {}
    for idx, key in enumerate(sorted_metrics):
        final_metrics_train[key] = metrics_train[key]
        final_metrics_test[key] = metrics_test[key]
        if idx + 1 == n:
            break

    print(f'\nChosen metrics: {len(final_metrics_train)}')

    model = training_regression(dt_train, final_metrics_train)
    test_regression(model, dt_test, final_metrics_test)

def main_experiment():
    #READING THE DATA
    dt_train, dt_test = readData.read_data()

    metrics_train = m.get_metrics(dt_train)
    metrics_test = m.get_metrics(dt_test)
    sorted_metrics = sort_metrics(dt_train, metrics_train)

    for n in range(2, 1+len(sorted_metrics.keys())):
        final_metrics_train = {}
        final_metrics_test = {}
        for idx, key in enumerate(sorted_metrics):
            final_metrics_train[key] = metrics_train[key]
            final_metrics_test[key] = metrics_test[key]
            if idx + 1 == n:
                break

        print(f'\nChosen metrics: {len(final_metrics_train)}')

        model = training_regression(dt_train, final_metrics_train)
        test_regression(model, dt_test, final_metrics_test)


if __name__ == '__main__':
    #final_experiment()
    main_experiment()
