import jieba
import joblib
from config import Config

conf = Config()

rf_model = joblib.load(conf.save_model + '/rf_model.pkl')
tfidf = joblib.load(conf.save_model + '/tfidf.pkl')


def predict(data: dict):
    """
    预测函数
    :param data: {"text":"文本"}
    :return: {"text":"文本","pred_class":"预测类别"}
    """

    words = " ".join(jieba.lcut(data["text"]))
    x = tfidf.transform([words])
    y_pred = rf_model.predict(x)
    data['pred_class'] = conf.class_list[y_pred[0]]

    return data


if __name__ == '__main__':
    data = predict({"text": "我和朋友在微信平台上，认识好友投资理财，我们被骗130000元，不知道能追回来吗？"})
    print(data)
