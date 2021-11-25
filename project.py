import pandas as pd
from nltk.metrics import jaccard_distance
import metrics as m
from scipy.stats import pearsonr
import numpy as np
import operator

def read_data():
    path = '/home/ferrando/master/IHLT-project/'

    #test_files = ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews']
    #train_files = ['MSRpar', 'MSRvid', 'SMTeuroparl']
    train_files = ['MSRvid','SMTeuroparl']
    test_files = ['MSRvid','SMTeuroparl','surprise.OnWN','surprise.SMTnews']
    dt_train = pd.read_csv(f'{path}train/STS.input.{train_files[0]}.txt',sep='\t',header=None)
    gs_train = pd.read_csv(f'{path}train/STS.gs.{train_files[0]}.txt',sep='\t',header=None)
    dt_test = pd.read_csv(f'{path}test-gold/STS.input.{test_files[0]}.txt', sep='\t', header=None)
    gs_test = pd.read_csv(f'{path}test-gold/STS.gs.{test_files[0]}.txt', sep='\t', header=None)
    for train_file,test_file in zip(train_files,test_files):
        if test_file !=test_files[0]: #si no es la primera iteracio
            new_dt_train = pd.read_csv(f'{path}train/STS.input.{train_file}.txt', sep='\t', header=None)
            new_gs_train = pd.read_csv(f'{path}train/STS.gs.{train_file}.txt', sep='\t', header=None)
            new_dt_test = pd.read_csv(f'{path}test-gold/STS.input.{test_file}.txt',sep='\t', header=None)
            new_gs_test = pd.read_csv(f'{path}test-gold/STS.gs.{test_file}.txt', sep='\t', header=None)

            #concatenate all the files in a single dataframe
            dt_train = pd.concat([dt_train, new_dt_train], axis=0, ignore_index=True)
            gs_train =  pd.concat([gs_train, new_gs_train], axis=0, ignore_index=True)
            dt_test = pd.concat([dt_test, new_dt_test], axis=0, ignore_index=True)
            gs_test = pd.concat([gs_test, new_gs_test], axis=0, ignore_index=True)

    dt_train = pd.concat([dt_train,gs_train], axis=1, ignore_index=True)
    dt_train.columns = [0, 1, 'gs']
    dt_test = pd.concat([dt_test,gs_test],axis=1, ignore_index=True)
    dt_test.columns = [0, 1, 'gs']

    return dt_train, dt_test


def jacc_sim(list1, list2):
    return 1-jaccard_distance(set(list1), set(list2)) # 1-distance


def training(dt_train, metrics, n=5):
    if n > len(metrics.values()):
        return metrics.keys()

    correlations = {}
    for metric in metrics:
        metrics[metric] = (metrics[metric] - np.min(metrics[metric])) / (np.max(metrics[metric]) - np.min(metrics[metric]))
        correlations[metric] = pearsonr(dt_train['gs'], metrics[metric])[0]
    
    sorted_c = dict(sorted(correlations.items(), key=operator.itemgetter(1),reverse=True))
    print(sorted_c)

    chosen_metrics = []
    for i in range(n):
        chosen_metrics.append(list(correlations.keys())[i])

    return chosen_metrics

def test(dt_test, metrics_test, chosen_metrics):
    metrics = []
    for i in chosen_metrics:
        metrics.append(metrics_test[i])
    
    prediction = np.average(metrics, axis=0)
    print('FINAL SCORE = {}'.format(pearsonr(dt_test['gs'], prediction)[0]))


if __name__ == '__main__':
    #READING THE DATA
    dt_train, dt_test = read_data()
    metrics_train = m.get_metrics(dt_train)
    # aixo ens ho podriem estalviar heavy, nomes caldria computar les X metriques millors
    metrics_test = m.get_metrics(dt_test)

    chosen_metrics = training(dt_train, metrics_train, n=2)
    test(dt_test, metrics_test, chosen_metrics)

