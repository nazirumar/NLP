import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from time import sleep
# Data collection and understanding

email_data = pd.read_csv('spam.csv', encoding='latin1')
# print(email_data.columns)
email_data = email_data[['v1', 'v2']]
email_data = email_data.rename(columns={'v1':'Target', 'v2':"Email"})

# Text processing and feature engineering
print("\n")
print("Data Preprocessing: ", end='')
for char in  '.' * 50:
    sleep(0.1)
    print(char, end='')
print("\n Display First (4) Spam Dataset: \n", email_data.head(4))

def text_data(email):
    #pre processing steps like lower case, stemming and lemmatization
    email.apply(lambda x: " ".join(x.lower() for x in x.split()))
    # print(email_data)
    stop = stopwords.words('english')
    email.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # print(email_data)
    st=PorterStemmer()
    email.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    # print(email_data.head(4))
    email.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    # print(email_data.head(4))
    return email

email_fun = text_data(email_data['Email'])

#Splitting data into train and validation

print("\n")
print("Split Data into Train and Validation: ", end='')
for char in  '.' * 50:
    sleep(0.1)
    print(char, end='')
print("\n")

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(email_fun, email_data['Target'], train_size=0.8, random_state=42)

# TFIDF feature generation for a maximum of 5000 features

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

tfidf = TfidfVectorizer(analyzer='word',
                        token_pattern=r'\w{1,}',
                        max_features=5000) # max
tfidf.fit(text_data(email_data['Email']))
x_train_tfidf = tfidf.transform(train_x)
x_valid_tfidf = tfidf.transform(valid_x)

# print(x_train_tfidf.data)

#  Model training
import pickle

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_nearal_net=False):
    model = classifier.fit(feature_vector_train, label)
    predicted = model.predict(feature_vector_valid)
    print("\n Train Model: ", end='')
    for char in  '.' * 50:
        sleep(0.1)
        print(char, end='')
    print("\n Display Predicted Value and Actual Value: \n")
    print("Predicted Dataset", predicted)
    print("Actual Dataset", valid_y)
    accuracy = (predicted == valid_y).mean()*100
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    # metrics.accuracy_score(predictio, valid_y)
    return accuracy
# Naïve Bayesian Classifier
accuracy = train_model(naive_bayes.MultinomialNB(alpha=0.2),
                       x_train_tfidf, train_y, x_valid_tfidf)
print("\n")
print("Model Accuracy: ", end='')
for char in  '.' * 50:
    sleep(0.1)
    print(char, end='')
print("\n")

print("Accuracy: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors

# accuracy = train_model(linear_model.LogisticRegression(), x_train_tfidf, train_y, x_valid_tfidf)
# print("Accuracy:", accuracy)


