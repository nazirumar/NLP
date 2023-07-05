from textblob import TextBlob

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from itertools import chain
from pywsd.lesk import simple_lesk


"""Problem
            Understanding disambiguating word sense."""
Text1 = 'I went to the bank to deposit my money'
Text2 = 'The river bank was full of dead fishes'

# sentences

bank_sents = ['I went to the bank to deposit my money', 'The river bank was full of dead fishes']

# calling the lesk function and printing results for both the sentences
print("Context-1", bank_sents[0])
answer = simple_lesk(bank_sents[0], 'bank')
print("Sense:", answer)
print("Definition : ", answer.definition())

print("Context-2", bank_sents[1])
answer = simple_lesk(bank_sents[1], 'bank', 'n')
print('Sense:', answer)
print("Definition : ", answer.definition())