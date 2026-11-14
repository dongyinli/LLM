'''
    config配置文件
'''
class Config():
    def __init__(self):
        self.train_datapath = '../data/train_data.tsv'
        self.test_datapath = '../data/test_data.tsv'
        self.val_datapath = '../data/val_data.tsv'

        self.process_train_datapath = './data/process_train.csv'
        self.process_test_datapath = './data/process_test.csv'
        self.process_val_datapath = './data/process_val.csv'

        self.stop_words_path = '../data/stopwords.txt'

        self.class_data_path = '../data/class.txt'
        with open(self.class_data_path, 'r', encoding='utf-8') as f:
            self.class_list = f.read().strip().split('\n')

        self.save_model = './save_model'
        self.save_predict = './result'