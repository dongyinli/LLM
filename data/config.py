class Config():
    def __init__(self):

        self.train_datapath = './train_data.tsv'

        self.test_datapath = './test_data.tsv'

        self.val_datapath = './val_data.tsv'

        self.class_path = './class.txt'

        with open(self.class_path, 'r', encoding='utf-8') as f:
            self.class_list = f.read().strip().split('\n')

if __name__ == '__main__':
    print(Config().class_list)
