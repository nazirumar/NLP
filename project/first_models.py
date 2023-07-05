
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import tokenize
import nltk
from textblob import TextBlob
from nltk.stem import PorterStemmer

text=['This is introduction to NLP','It is likely to be useful, to people ','Machine learning is the new electrcity','Therewould be less hype around AI and more action going forward','python is the best tool!','R is good langauage','I like this book','I want more books like this']

df = pd.DataFrame({'tweet':text})

# Execute Lower() function on the text data
df['tweet'] = df['tweet'].apply(lambda x: " ".join(x.lower()))

# Remove punctuation
df['tweet'] = df['tweet'].apply(lambda x:' '.join([i for i in x if i not in string.punctuation]))

# Remove stopwords
stop_words=set(stopwords.words('english'))  # set of all english words
df['tweet']=df['tweet'].apply(lambda x :' '.join(word for word in x.split() if x not in stop_words))

print(df)
# textblob
TextBlob(df['tweet'][3]).words # sentiment analysis using text blob

# Tokenize into individual tokens (lemmatization is done automatically by NLTK library).
tokens=[nltk.word_tokenize(t) for t in df['tweet']]

# stem
# ps = PorterStemmer()
# for token in tokens[0]:
#     print("Original Word:",token," Stemmed Word", ps.stem(token))

# print(tokens)
