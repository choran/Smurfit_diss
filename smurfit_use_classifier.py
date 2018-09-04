import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import json
import seaborn as sns
import keras.layers as layers
from keras.models import Model
from keras import backend as K

np.random.seed(10)
import argparse
path = "/home"
# Setup command line so that you can enter a new question
parser = argparse.ArgumentParser(description='Check if the classifer can identify Intercom questions')
parser.add_argument('-q', dest='question', action='store',
                    nargs='*', help='Test input question')
parser.add_argument('-b', dest='baseline', action='store', type=int,
                    help='Use baseline questions')
parser.add_argument('-t', dest='train', action='store', type=int,
                    help='Train the classifier')
args = parser.parse_args()

# Import the Universal Sentence Encoder's TF Hub module
module_url = path + "/use_module/"
embed = hub.Module(module_url, trainable=True)


def get_dataframe(filename):
    lines = open(filename, 'r').read().splitlines()
    data = []
    for i in range(0, len(lines)):
        label = lines[i].split(' ')[0]
        label = label.split(":")[0]
        text = ' '.join(lines[i].split(' ')[1:])
        text = re.sub('[^A-Za-z0-9 ,\?\'\"-._\+\!/\`@=;:]+', '', text)
        data.append([0, text])

    df = pd.DataFrame(data, columns=['label', 'text'])
    df.label = df.label.astype('category')
    return df

def get_baseline():
    # Read in the list of qustions from the file
    qs_file = 'BASE LINE FILE'
    base_qs = pd.read_csv(qs_file)
    queries = [q for q in base_qs['Question']]
    return(queries)

def get_intercom_df(path):
    # Get the file of questions
    with open(path) as fd:
        answers = [json.loads(line) for line in fd]
    data = []
    # First iterate through the lists of questions
    for qs in answers:
        text = qs['query']
        data.append([1, text])
    df = pd.DataFrame(data, columns=['label', 'text'])
    df.label = df.label.astype('category')
    return df


df_train_non = get_dataframe('train_5500.txt')
# print(df_train_non.head())
df_train_intercom = get_intercom_df('intercom_train.jsonl')
# print(df_train_intercom.head())
df_train = df_train_non.append(df_train_intercom, ignore_index=True)
# print(df_train)

category_counts = 2


def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = layers.Lambda(UniversalEmbedding, output_shape=(512,))(input_text)
dense = layers.Dense(256, activation='relu')(embedding)
pred = layers.Dense(category_counts, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

train_text = df_train['text'].tolist()
train_text = np.array(train_text, dtype=object)[:, np.newaxis]

train_label = np.asarray(pd.get_dummies(df_train.label), dtype=np.int8)

# print(train_text[-5:], train_label[-5:])
# print(train_text[:5], train_label[:5])

df_test_non = get_dataframe('test_data.txt')
df_test_intercom = get_intercom_df('intercom_test.jsonl')
df_test = df_test_non.append(df_test_intercom, ignore_index=True)

test_text = df_test['text'].tolist()
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = np.asarray(pd.get_dummies(df_test.label), dtype=np.int8)

if (args.train):
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        history = model.fit(train_text,
                            train_label,
                            validation_data=(test_text, test_label),
                            epochs=10,
                            batch_size=32)
        model.save_weights('./trainable_model.h5')

if (args.train) is None:
    results = []
    if (args.baseline):
        new_text = get_baseline()
    else:
        new_text = args.question
    print(new_text)
    # new_text = ["In what year did the titanic sink ?",
    #        "What is the highest peak in California ?",
    #        "Who invented the light bulb ?"]
    new_text = np.array(new_text, dtype=object)[:, np.newaxis]
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        model.load_weights('./trainable_model.h5')
        predicts = model.predict(new_text, batch_size=32)
        eval = model.evaluate(test_text, test_label, batch_size=32)
        print(eval)

    # categories = df_train.label.cat.categories.tolist()
    categories = [0, 1]
    predict_logits = predicts.argmax(axis=1)
    predict_labels = [categories[logit] for logit in predict_logits]
    for q, l in zip(new_text, predict_labels):
        print("%s: %s" % (q, l))
        res = "Intercom question" if l else "Not an Intercom question"
        results.append([q, res])
    df = pd.DataFrame(results, columns=['Question', 'Prediction'])
    df.sort_values('Prediction', ascending=False)
    df.to_csv('use_trainable_redict.csv')
    