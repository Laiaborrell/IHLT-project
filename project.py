import readData
from nltk.metrics import jaccard_distance
import metrics as m
from scipy.stats import pearsonr
import numpy as np
import operator

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
    dt_train, dt_test = readData.read_data()
    metrics_train = m.get_metrics(dt_train)

    # aixo ens ho podriem estalviar heavy, nomes caldria computar les X metriques millors
    metrics_test = m.get_metrics(dt_test)

    chosen_metrics = training(dt_train, metrics_train, n=2)
    test(dt_test, metrics_test, chosen_metrics)

