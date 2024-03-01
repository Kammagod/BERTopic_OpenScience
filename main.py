from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import os

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ['CURL_CA_BUNDLE'] = ''

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']


topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

print("aaa")
print(topic_model.get_topic_info())