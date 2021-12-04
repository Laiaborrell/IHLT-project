import pandas as pd
import regex as re

def load_dataframe(input_filepath):
  current_file_path = input_filepath
  try:
    # Read inputs
    data = []
    with open(input_filepath, 'r',encoding='utf') as f:
      lines = f.read().splitlines()
      for line in lines:
        data.append(line.split("\t"))
    df = pd.DataFrame(data, columns = [0, 1])
    # Read Gold Standard
    current_file_path = re.sub("input", "gs", input_filepath)
    df["gs"] = pd.read_csv(current_file_path, sep='\t', header=None)

  except Exception as e:
    raise Exception(f"ERROR while reading {current_file_path}:\n\t{e}")

  return df
    

def read_data():
    #path = '/home/ferrando/master/IHLT-project/'
    path = ''
    test_files = ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', 'surprise.SMTnews']
    train_files = ['MSRpar', 'MSRvid', 'SMTeuroparl']
   
    dt_train = load_dataframe(f'{path}train/STS.input.{train_files[0]}.txt')
    dt_test = load_dataframe(f'{path}test-gold/STS.input.{test_files[0]}.txt')
  
    for train_file,test_file in zip(train_files,test_files):
        if test_file !=test_files[0]: #si no es la primera iteracio
            new_dt_train = load_dataframe(f'{path}train/STS.input.{train_file}.txt')
            new_dt_test = load_dataframe(f'{path}test-gold/STS.input.{test_file}.txt')

            #concatenate all the files in a single dataframe
            dt_train = pd.concat([dt_train, new_dt_train], axis=0, ignore_index=True)
            dt_test = pd.concat([dt_test, new_dt_test], axis=0, ignore_index=True)

    dt_train.columns = [0, 1, 'gs']
    dt_test.columns = [0, 1, 'gs']

    return dt_train, dt_test