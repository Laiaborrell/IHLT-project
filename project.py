import pandas as pd

def read_data():
    #files = ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews']
    files = ['MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews']
    dt = pd.read_csv(f'test-gold/STS.input.{files[0]}.txt',sep='\t',header=None)
    for file in files:
        if file !=files[0]:
            new_dt = pd.read_csv(f'test-gold/STS.input.{file}.txt',sep='\t',header=None)
            #concatenate all the files in a single dataframe
            dt = pd.concat([dt,new_dt],axis=0)
    print(dt.shape)
    print(dt.head())



if __name__ == '__main__':
    read_data()
