from pandas.core.frame import DataFrame
import readData
from nltk.metrics import jaccard_distance
import metrics as m
from scipy.stats import pearsonr
import numpy as np
import operator
from sklearn import linear_model
from sklearn.feature_selection import SequentialFeatureSelector
import multiprocessing
import matplotlib.pyplot as plt
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

def sort_metrics(title, dt_train, metrics):
    metrics = preprocess_metrics(metrics)

    correlations = {}
    for metric in metrics:
        correlations[metric] = pearsonr(dt_train['gs'], metrics[metric])[0]
    sorted_c = dict(sorted(correlations.items(), key=operator.itemgetter(1), reverse=True))
    printTopMetrics(sorted_c)

    fig = plt.figure(figsize=(10,10))
    plt.title(title,fontsize=16)
    plt.ylabel('Individual Pearson correlation with gs',fontsize=14)
    plt.bar(correlations.keys(), correlations.values())
    plt.xticks(rotation=45,fontsize=14)
    fig.savefig(title +'.png')

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

def main_experiment(file_object, lexical=False, syntactic=False, all_metrics=True, postprocess=False, distance='jaccard', stop_words=False):
    experiment = f'\nExperiment with lexical={lexical}, syntactic={syntactic}, allmetrics={all_metrics}, postprocess={postprocess} and stop_words={stop_words}.'
    print(experiment)
    file_object.write(experiment + '\n')
    dt_train, dt_test = readData.read_data()
    metrics_train = m.get_metrics(dt_train, lexical=lexical, syntactic=syntactic, all_metrics=all_metrics, distance=distance, stop_words=stop_words)
    metrics_test = m.get_metrics(dt_test, lexical=lexical, syntactic=syntactic, all_metrics=all_metrics, distance=distance, stop_words=stop_words)

    sort_metrics(f'lexical={lexical}, syntactic={syntactic}, allmetrics={all_metrics}, postprocess={postprocess}, stop_words={stop_words}', 
        dt_train, metrics_train)
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
        chosen_metrics.append(sfs.get_support())

        result = f'\tChosen metrics: {X_train.shape[1]}, Pearson correlation: {p_correlation}'
        print(result)
        file_object.write(result + '\n')
        final_metrics = ''
        for idx, metric in enumerate(chosen_metrics[i-1]):
            if metric:
                final_metrics = final_metrics + list(metrics_train.keys())[idx] + ', '
        file_object.write(final_metrics + '\n')

    # All metrics
    model = training_regression(dt_train, X)
    p_correlation = test_regression(model, dt_test, X_test, postprocess=postprocess)
    results.append(p_correlation)
    chosen_metrics.append(X.keys())
    result = f'\tChosen metrics: {X.shape[1]}, Pearson correlation: {p_correlation}'
    print(result)
    file_object.write(result + '\n')
    final_metrics = ''
    for idx, metric in enumerate(chosen_metrics[i]):
        if metric:
            final_metrics = final_metrics + list(metrics_train.keys())[idx] + ', '
    file_object.write(final_metrics + '\n')

    max_index = results.index(max(results))
    final_result = f'\nBEST SETUP: {max_index+1} metrics, Correlation={results[max_index]}'
    final_metrics = ''
    for i, metric in enumerate(chosen_metrics[max_index]):
        if metric:
            final_metrics = final_metrics + list(metrics_train.keys())[i] + ', '
    print(final_result)
    print(final_metrics)
    file_object.write(final_result + '\n')
    file_object.write(final_metrics + '\n')

    fig = plt.figure()
    plt.title(f'lexical={lexical}, syntactic={syntactic}, allmetrics={all_metrics}, postprocess={postprocess}, stop_words={stop_words}')
    plt.xlabel('Number of selected metrics')
    plt.ylabel('Pearson correlation with gs')
    plt.plot(range(1,len(results)+1), results)
    fig.savefig(f'plots/lexical={lexical}_syntactic={syntactic}_allmetrics={all_metrics}_postprocess={postprocess}_stop_words={stop_words}.png')

if __name__ == '__main__':
    if os.path.exists(FILE):
        os.remove(FILE)

    file_object = open(FILE, 'w+')

    #final_experiment()
    main_experiment(file_object, lexical=True, syntactic=False, all_metrics=False, postprocess=False, distance='jaccard', stop_words=False)
    main_experiment(file_object, lexical=True, syntactic=False, all_metrics=False, postprocess=True, distance='jaccard', stop_words=False)
    main_experiment(file_object, lexical=True, syntactic=False, all_metrics=False, postprocess=False, distance='jaccard', stop_words=True)
    main_experiment(file_object, lexical=True, syntactic=False, all_metrics=False, postprocess=True, distance='jaccard', stop_words=True)
    
    main_experiment(file_object, lexical=False, syntactic=True, all_metrics=False, postprocess=False, distance='jaccard', stop_words=False)
    main_experiment(file_object, lexical=False, syntactic=True, all_metrics=False, postprocess=True, distance='jaccard', stop_words=False)
    main_experiment(file_object, lexical=False, syntactic=True, all_metrics=False, postprocess=False, distance='jaccard', stop_words=True)
    main_experiment(file_object, lexical=False, syntactic=True, all_metrics=False, postprocess=True, distance='jaccard', stop_words=True)
    
    main_experiment(file_object, lexical=True, syntactic=True, all_metrics=False, postprocess=False, distance='jaccard', stop_words=False)
    main_experiment(file_object, lexical=True, syntactic=True, all_metrics=False, postprocess=True, distance='jaccard', stop_words=False)
    main_experiment(file_object, lexical=True, syntactic=True, all_metrics=False, postprocess=False, distance='jaccard', stop_words=True)
    main_experiment(file_object, lexical=True, syntactic=True, all_metrics=False, postprocess=True, distance='jaccard', stop_words=True)
    
    main_experiment(file_object, lexical=True, syntactic=True, all_metrics=True, postprocess=False, distance='jaccard', stop_words=False)
    main_experiment(file_object, lexical=True, syntactic=True, all_metrics=True, postprocess=True, distance='jaccard', stop_words=False)
    main_experiment(file_object, lexical=True, syntactic=True, all_metrics=True, postprocess=False, distance='jaccard', stop_words=True)
    main_experiment(file_object, lexical=True, syntactic=True, all_metrics=True, postprocess=True, distance='jaccard', stop_words=True)
    
    file_object.close()
    