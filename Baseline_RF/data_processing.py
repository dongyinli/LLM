import pandas as pd
import jieba

from config import Config

conf = Config()


def data_processing(read_path, save_path):

    data = pd.read_csv(read_path, sep='\t', names=['content', 'label'])
    data['words'] = data['content'].apply(lambda x: ' '.join(jieba.lcut(x)))
    data.to_csv(save_path, sep='\t', index=False)


def main():
    data_processing(conf.train_datapath, conf.process_train_datapath)
    data_processing(conf.test_datapath, conf.process_test_datapath)
    data_processing(conf.val_datapath, conf.process_val_datapath)


if __name__ == '__main__':
    main()
