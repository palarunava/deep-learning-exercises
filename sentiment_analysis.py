import re
import pandas as pd
#import nltk
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Embedding

import spacy
import numpy as np

#%%
# This function removes the <br> tags, the '"' double quotes, '\' slashes, etc. and cleans the text.
def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)
 
    # Strip escaped quotes
    text = text.replace('\\"', '')
 
    # Strip quotes
    text = text.replace('"', '')
 
    return text
 
#%%
df = pd.read_csv('.\data\IMDB_Movie_Reviews\labeledTrainData.tsv', sep='\t', quoting=3)
# Introduce a new column in the existing data structure and populate the cleaned text into it.
df['cleaned_review'] = df['review'].apply(clean_review)
# Split the data into train and test sets using train_test_split function
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2)

#%%
"""
1. stop_words: Since CountVectorizer just counts the occurrences of each word in its vocabulary, extremely common words like ‘the’, ‘and’, etc. will become very important features while they add little meaning to the text. Your model can often be improved if you don’t take those words into account. Stop words are just a list of words you don’t want to use as features. You can set the parameter stop_words=’english’ to use a built-in list. Alternatively you can set stop_words equal to some custom list. This parameter defaults to None.
2. ngram_range: An n-gram is just a string of n words in a row. E.g. the sentence ‘I am Groot’ contains the 2-grams ‘I am’ and ‘am Groot’. The sentence is itself a 3-gram. Set the parameter ngram_range=(a,b) where a is the minimum and b is the maximum size of ngrams you want to include in your features. The default ngram_range is (1,1). In a recent project where I modeled job postings online, I found that including 2-grams as features boosted my model’s predictive power significantly. This makes intuitive sense; many job titles such as ‘data scientist’, ‘data engineer’, and ‘data analyst’ are 2 words long.
3. min_df, max_df: These are the minimum and maximum document frequencies words/n-grams must have to be used as features. If either of these parameters are set to integers, they will be used as bounds on the number of documents each feature must be in to be considered as a feature. If either is set to a float, that number will be interpreted as a frequency rather than a numerical limit. min_df defaults to 1 (int) and max_df defaults to 1.0 (float).
4. max_features: This parameter is pretty self-explanatory. The CountVectorizer will choose the words/features that occur most frequently to be in its’ vocabulary and drop everything else.
https://medium.com/@rnbrown/more-nlp-with-sklearns-countvectorizer-add577a0b8c8

max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:
max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
max_df = 25 means "ignore terms that appear in more than 25 documents".
The default max_df is 1.0, which means "ignore terms that appear in more than 100% of the documents". Thus, the default setting does not ignore any terms.

min_df is used for removing terms that appear too infrequently. For example:
min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
min_df = 5 means "ignore terms that appear in less than 5 documents".
The default min_df is 1, which means "ignore terms that appear in less than 1 document". Thus, the default setting does not ignore any terms.
"""
#nltk.download()
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), lowercase=True, min_df=3, max_df=0.9, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train)

#%%
model = Sequential()
 
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#%%
model.fit(X_train_onehot[:-100], y_train[:-100], epochs=2, batch_size=128, verbose=1, validation_data=(X_train_onehot[-100:], y_train[-100:]))

#%%
scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print("Accuracy:", scores[1])  # Accuracy: 0.875

#%%
# Prepares a dictionary with all the features extracted (5000) by the CountVectorizer
word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()
 
#%%
# Preprocesses and tokenizes the input string and returns the index numbers of the tokenized terms from the dictionary.
def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes
 
#%%
# Tokenizes a text and converts it into a sequence of indices corresponding to every word in the dictionary.
print(to_sequence(tokenize, preprocess, word2idx, "This is an important test!"))  # [2269, 4453]
X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_train]
print(X_train_sequences[0])

#%%
# Compute the max lenght of a text
MAX_SEQ_LENGHT = len(max(X_train_sequences, key=len))
print("MAX_SEQ_LENGHT=", MAX_SEQ_LENGHT)

#%%
# Make all sequnces to be as long as the the longest sequence by padding the remaining spaces with a value (here it is the number of features extracted from the text corpus (5000))
N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)
print(X_train_sequences[0])

#%%
model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGHT))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#%%
model.fit(X_train_sequences[:-100], y_train[:-100], epochs=3, batch_size=512, verbose=1, validation_data=(X_train_sequences[-100:], y_train[-100:]))

#%%
X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)

#%%
scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1]) # 0.8766

#%%
model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGHT))
model.add(LSTM(64))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#%%
model.fit(X_train_sequences[:-100], y_train[:-100], epochs=2, batch_size=128, verbose=1, validation_data=(X_train_sequences[-100:], y_train[-100:]))

#%%
scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1]) # 0.875

#%%
#python -m spacy download en              # default English model (~50MB)
#python -m spacy download en_core_web_md  # larger English model (~1GB)
nlp = spacy.load('en_core_web_md')
 
EMBEDDINGS_LEN = len(nlp.vocab['apple'].vector)
print("EMBEDDINGS_LEN=", EMBEDDINGS_LEN)  # 300
 
# Create a matrix of size (number of features extracted from text corpus, number of spacy features for each feature). Here size = (5000, 300).
embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, EMBEDDINGS_LEN))
for word, idx in word2idx.items():
    try:
        embedding = nlp.vocab[word].vector
        embeddings_index[idx] = embedding
    except:
        pass

#%%
model = Sequential()
# In Embedding layer, the first and second argument are the input and output dimensions. The dimension of the learned weights should match (input_dim, output_dim).
# The length of the input sequence is the length of the padded sequence which is the length of the longest text sequence.
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    EMBEDDINGS_LEN,  # Embedding size
                    weights=[embeddings_index],
                    input_length=MAX_SEQ_LENGHT,
                    trainable=False))
model.add(LSTM(300))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#%%
model.fit(X_train_sequences[:-100], y_train[:-100], epochs=1, batch_size=128, verbose=1, validation_data=(X_train_sequences[-100:], y_train[-100:]))

#%%
scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1])  # 0.8508

#%%
GLOVE_PATH = './data/glove.6B/glove.6B.50d.txt'
GLOVE_VECTOR_LENGHT = 50
 
def read_glove_vectors(path, lenght):
    embeddings = {}
    with open(path, encoding='utf8') as glove_f:
        for line in glove_f:
            chunks = line.split()
            assert len(chunks) == lenght + 1
            embeddings[chunks[0]] = np.array(chunks[1:], dtype='float32')
 
    return embeddings
 
#%%
GLOVE_INDEX = read_glove_vectors(GLOVE_PATH, GLOVE_VECTOR_LENGHT)
 
# Init the embeddings layer with GloVe embeddings
embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, GLOVE_VECTOR_LENGHT))
for word, idx in word2idx.items():
    try:
        embedding = GLOVE_INDEX[word]
        embeddings_index[idx] = embedding
    except:
        pass

#%%
model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    GLOVE_VECTOR_LENGHT,  # Embedding size
                    weights=[embeddings_index],
                    input_length=MAX_SEQ_LENGHT,
                    trainable=False))
model.add(LSTM(128))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
 
#%%
model.fit(X_train_sequences[:-100], y_train[:-100], 
          epochs=3, batch_size=128, verbose=1, 
          validation_data=(X_train_sequences[-100:], y_train[-100:]))
 
#%%
scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1])  # 0.8296