from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
#vectorizer = CountVectorizer()
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), lowercase=True, min_df=3, max_df=0.9, max_features=5000)
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
