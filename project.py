import nltk
nltk.download('punkt')
import pandas as pd
import treetaggerwrapper as ttpw
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
            new_dt_test = pd.read_csv(f'test-gold/STS.input.{test_file}.txt',sep='\t',header=None)
            new_gs_test = pd.read_csv(f'test-gold/STS.gs.{test_file}.txt', sep='\t', header=None)

            #concatenate all the files in a single dataframe
            dt_train = pd.concat([dt_train,new_dt_train],axis=0)
            gs_train =  pd.concat([gs_train,new_gs_train],axis=0)
            dt_test = pd.concat([dt_test, new_dt_test], axis=0)
            gs_test = pd.concat([gs_test, new_gs_test], axis=0)

    dt_train = pd.concat([dt_train,gs_train],axis=1)
    dt_train.columns = [0,1,'gs']
    dt_test = pd.concat([dt_test,gs_test],axis=1)
    dt_test.columns = [0, 1, 'gs']

    print(dt_train.head())
    print(dt_test.head())
    return dt_train,dt_test


def jacc_sim(list1, list2):
    return 1-jaccard_distance(set(list1), set(list2)) # 1-distance


def lemmatize(tagger, text):
    tags = tagger.tag_text(text)
    lemmas = [t.split('\t')[-1] for t in tags]
    return lemmas


def training(dt_train, gs_train, metrics, n=5):
    if n > len(metrics):
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


# Measures TODO:
# Longest Common Substring
# Longest Common Subsequence
# Greedy String Tiling
# Character n-grams n=1,2,3,4
# Words n-grams n=1,2,3,4


if __name__ == '__main__':
    #READING THE DATA
    dt_train,dt_test = read_data()
    r,c = dt_train.shape
    metrics = {}
    
    #tagger = ttpw.TreeTagger(TAGLANG='en')
    a_lems, b_lems, js_l = [], [], []
    a_words, b_words, js_w = [], [], []
    for i in range(r): # iteration over the rows
        words_a, words_b = (nltk.word_tokenize(dt_train[0][i]) , nltk.word_tokenize(dt_train[1][i]))
        #lemmas_a, lemmas_b = (lemmatize(tagger, dt_train[0][i]) , lemmatize(tagger, dt_train[1][i]))
        #a_lems.append(lemmas_a)
        #b_lems.append(lemmas_b)
        a_words.append(words_a)
        b_words.append(words_b)
        #js_l = jacc_sim(lemmas_a, lemmas_b)
        js_w = jacc_sim(words_a, words_b)
    
    dt_train['words_a'] = a_words
    dt_train['words_b'] = b_words
    dt_train['lemmas_a'] = a_lems
    dt_train['lemmas_b'] = b_lems
    metrics['lemmas_js'] = js_l
    metrics['words_js'] = js_w

    # Metrics loop
    for i in range(r):
        metrics['lc_substring'] = m.lc_substring(dt_train[0][i], dt_train[1][i])
        metrics['lc_subsequence'] = m.lc_subsequence(dt_train[0][i], dt_train[1][i])
        for j in range(1,4):
            w_metric_name = 'w_ngrams_'+i
            c_metric_name = 'c_ngrams_'+i
            metrics[w_metric_name] = m.compare_words_ngrams(dt_train[0][i], dt_train[1][i], j)
            metrics[c_metric_name] = m.compare_character_ngrams(dt_train[0][i], dt_train[1][i], j)

    training(dt_train, metrics, 2)
