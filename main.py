

from collections import Counter
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
import tensorflow_datasets as tfds
import re

#preprocessing
import string
import nltk
from nltk.corpus import stopwords


df = pd.read_csv("SPAM text message 20170820 - Data.csv")

#print(df.head(2))
#print(df.describe())

def preprocessing(s):
    s = s.lower()
    s = re.sub(r"[^0-9a-z]", " ", s)
    s = re.sub(r"\s{2,}", " ", s)

    valid_words = [w for w in s.split() if w not in (string.punctuation and stopwords.words("english"))]
    stemmed_words = " ".join([PorterStemmer().stem(w) for w in valid_words])

    return stemmed_words


print(preprocessing("Hello, World! This is a SPAM      message!"))


vectorizer = TfidfVectorizer()
le = LabelEncoder()
X = df.Message.apply(lambda s: preprocessing(s))
X = vectorizer.fit_transform(X)
y = le.fit_transform(df.Category)
le.fit_transform(["ham", "spam"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)


preds = naive_bayes.predict(X_test)

print(classification_report(y_test, preds))

rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

print(classification_report(y_test, preds))
le = LabelEncoder()
df["Category_enc"] = le.fit_transform(df.Category)
df.head()
X = df.Message
y = df.Category_enc

dataset = tf.data.Dataset.from_tensor_slices((X, y))


dataset = dataset.shuffle(6000, reshuffle_each_iteration=False)
ds_test = dataset.take(1000)
ds_train = dataset.skip(1000).take(4500)

tokenizer = tfds.features.text.Tokenizer()
token_counts = Counter()
for example in ds_train:
    tokens = tokenizer.tokenize(example[0].numpy())
    token_counts.update(tokens)

encoder = tfds.features.text.TokenTextEncoder(token_counts)


example_str = encoder.encode("This is a spam")
example_str


def encode(text_tensor, label):
    text = text_tensor.numpy()
    encoded_text = encoder.encode(text)

    return encoded_text, label


def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


ds_train = ds_train.map(encode_map_fn)
ds_test = ds_test.map(encode_map_fn)



for example in ds_train.shuffle(4500).take(5):
    print(f"Sequence length: {example[0].shape}")



train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
test_data = ds_train.padded_batch(32, padded_shapes=([-1], []))


embedding_dim = 20
vocab_size = len(token_counts) + 2

tf.random.set_seed(1)

lstm = tf.keras.Sequential()
lstm.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embedding_layer"))
lstm.add(tf.keras.layers.LSTM(units=64, return_sequences=True, name="lstm_layer"))
lstm.add(tf.keras.layers.Dense(64, activation="relu"))
lstm.add(tf.keras.layers.Dense(1, activation="sigmoid"))


lstm.summary()


lstm.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])
lstm.fit(train_data, validation_data=test_data, epochs=20)


embedding_dim = 20
vocab_size = len(token_counts) + 2

tf.random.set_seed(1)

gru = tf.keras.Sequential()
gru.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embedding_layer"))
gru.add(tf.keras.layers.GRU(units=64, return_sequences=True, name="gru_layer"))
gru.add(tf.keras.layers.Dense(64, activation="relu"))
gru.add(tf.keras.layers.Dense(1, activation="sigmoid"))



gru.summary()

gru.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])
gru.fit(train_data, validation_data=test_data, epochs=20)




