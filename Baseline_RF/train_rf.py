import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import Config

conf = Config()

train_data = pd.read_csv(conf.process_train_datapath, sep='\t')
test_data = pd.read_csv(conf.process_test_datapath, sep='\t')

x_train = train_data['words']
y_train = train_data['label']

x_test = test_data['words']
y_test = test_data['label']

with open(conf.stop_words_path, 'r', encoding='utf-8') as f:
    stop_words = f.read().strip().split()

tfidf = TfidfVectorizer(stop_words=stop_words)

x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)


model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15,
    # verbose=3,
    random_state=22
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'accuracy-->{accuracy_score(y_test, y_pred)}')
print(f'precision-->{precision_score(y_test, y_pred, average="weighted")}')
print(f'recall-->{recall_score(y_test, y_pred, average="weighted")}')
print(f'f1_score-->{f1_score(y_test, y_pred, average="weighted")}')
print(f'confusion_matrix-->\n{confusion_matrix(y_test, y_pred)}')

joblib.dump(model, conf.save_model + '/rf_model.pkl')
joblib.dump(tfidf, conf.save_model + '/tfidf.pkl')
