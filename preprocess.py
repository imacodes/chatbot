import nltk

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem.lancaster import LancasterStemmer
import numpy
import json
import pickle

stemmer = LancasterStemmer()

# The block of code loads the training data and store it in a variable
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    # Blank lists setup to store all of the patterns and which class/tag they belong to
    # List below is also used to store all the unique words in our patterns
    words = []
    labels = []
    docs_x = []
    docs_y = []
    # Created a loop to  through our JSON data and extract the data we want
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # For each pattern we will turn it into a list of words using nltk.word_tokenizer
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            #  Each pattern is added into our docs_x list
            docs_x.append(wrds)
            # Associated tag in docs_x is added into the docs_y list
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # This code creates a unique list of stemmed words to use in the next step of our data preprocessing
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
