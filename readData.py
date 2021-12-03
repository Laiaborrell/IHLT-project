import pandas as pd

def read_data():
    path = '/home/ferrando/master/IHLT-project/'
    path = ''
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