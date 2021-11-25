import pandas as pd
from nltk.metrics import jaccard_distance
import metrics as m
from scipy.stats import pearsonr


def read_data():
    #test_files = ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews']
    #train_files = ['MSRpar', 'MSRvid', 'SMTeuroparl']
    train_files = ['MSRvid','SMTeuroparl']
    test_files = ['MSRvid','SMTeuroparl','surprise.OnWN','surprise.SMTnews']
    dt_train = pd.read_csv(f'train/STS.input.{train_files[0]}.txt',sep='\t',header=None)
    gs_train = pd.read_csv(f'train/STS.gs.{train_files[0]}.txt',sep='\t',header=None)
    dt_test = pd.read_csv(f'test-gold/STS.input.{test_files[0]}.txt', sep='\t', header=None)
    gs_test = pd.read_csv(f'test-gold/STS.gs.{test_files[0]}.txt', sep='\t', header=None)
    for train_file,test_file in zip(train_files,test_files):
        if test_file !=test_files[0]: #si no es la primera iteracio
            new_dt_train = pd.read_csv(f'train/STS.input.{train_file}.txt', sep='\t', header=None)
            new_gs_train = pd.read_csv(f'train/STS.gs.{train_file}.txt', sep='\t', header=None)
            new_dt_test = pd.read_csv(f'test-gold/STS.input.{test_file}.txt',sep='\t', header=None)
            new_gs_test = pd.read_csv(f'test-gold/STS.gs.{test_file}.txt', sep='\t', header=None)

            #concatenate all the files in a single dataframe
            dt_train = pd.concat([dt_train, new_dt_train], axis=0, ignore_index=True)
            gs_train =  pd.concat([gs_train, new_gs_train], axis=0, ignore_index=True)
            dt_test = pd.concat([dt_test, new_dt_test], axis=0, ignore_index=True)
            gs_test = pd.concat([gs_test, new_gs_test], axis=0, ignore_index=True)

    dt_train = pd.concat([dt_train,gs_train], axis=1, ignore_index=True)
    dt_train.columns = [0, 1, 'gs']
    dt_test = pd.concat([dt_test,gs_test],axis=1, ignore_index=True)
    dt_test.columns = [0, 1, 'gs']

    #print(dt_train.head())
    #rint(dt_test.head())
    return dt_train, gs_train, dt_test, gs_test


def jacc_sim(list1, list2):
    return 1-jaccard_distance(set(list1), set(list2)) # 1-distance


def training(gs_train, metrics, n=5):
    if n > len(metrics.values()):
        return metrics.keys()

    correlations = {}
    for metric in metrics:
        correlations[metric] = pearsonr(gs_train, metrics[metric])[0]
    
    sorted_c = {k: v for k, v in sorted(correlations.items(), key=lambda item: item[1])}
    print(sorted_c)

    chosen_metrics = []
    for i in range(n):
        chosen_metrics.append(sorted_c.values()[i])

    return chosen_metrics

def test(df_test, metrics):
    gs = pd.read_csv('./train/STS.gs.SMTeuroparl.txt',sep='\t',header=None) # gold standard measure
    print('Comparing lemmas, jaccard distance = {}'.format(pearsonr(gs, dt['js lemmas'])[0]))


if __name__ == '__main__':
    #READING THE DATA
    dt_train, gs_train, dt_test, gs_test = read_data()
    metrics_train = m.get_metrics(dt_train)
    
    chosen_metrics = training(gs_train, metrics_train, n=2)

