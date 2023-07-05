from nltk.corpus import stopwords

def stopworldlist():

    stopworldlist = stopwords.words('english')
    for s in stopworldlist:
        with open('stopwords.txt', 'a+') as f:
            f.write('{}\n'.format(s))
            f.close()
        print(s)

def customizedstopwordremove():
    stop_words = set(['hi', 'bye'])
    line = """hi this is foo. bye"""

    print("".join(word for word in line.split() if word not in stop_words))


if __name__ == '__main__':
    # stopworldlist()
    customizedstopwordremove()