import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from nltk.parse.stanford import StanfordDependencyParser

text = "it's going to have the same format as exams 1 and 2 it may be a little bit longer Normally" \
       "your exams are 80 minutes So this may be a two hour exam or maybe even more."

print("NLTK:")

# test NLTK
# -------------------------------   NLTK    -------------------------------------------------------------+
#  Preprocess text
default_stopwords = set(nltk.corpus.stopwords.words('english'))
words = ""
def preprocess_NLTK(text):
    words = word_tokenize(text)  # wolds tokenization
    print("word_tokenize", words, "\n")
    words = [word for word in words if word not in default_stopwords]
    print("stopwords_removal", words, "\n")
    words = [word for word in words if len(word) > 1]
    print("punct_removal", words, "\n")
    words = [word for word in words if not word.isnumeric()]
    print("digit_removal", words, "\n")
    return words

words = preprocess_NLTK(text)
fdist = nltk.FreqDist(words)

#for word, frequency in fdist.most_common(3):
#    print(u'{};{}'.format(word, frequency))
tag = nltk.pos_tag(words)  # Tag part of speech to each  with
#print(tag)

ne_tree = nltk.ne_chunk(tag, binary=False)
#print(ne_tree)
#ne_tree.draw()  # draw a tree for the tagging

# filter for the pattern that personal pronoun followed with a verb
pattern = r"""pattern: {<PRP><VB.?>}"""
cp = nltk.RegexpParser(pattern)
tree = cp.parse(tag)
#tree.draw()
#for subtree in tree.subtrees(filter=lambda t: t.label() == 'pattern'):
#    print(subtree)

print("spaCy:")
# Test spaCy
# -------------------------------   spaCy    -------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")  # Load language model
doc = nlp(text)

#spacy.displacy.serve(doc, style="dep")  # draw the dependence tree, and it's shown in "http://localhost:5000"

tokenization = [token.text for token in doc]
print("tokenization: ", tokenization, "\n")
stop_removal = [token.text for token in doc if token.is_stop != True]
print("stop_removal: ", stop_removal, "\n")
punct_removal = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
print("punct_removal: ", punct_removal, "\n")
digit_removal = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.is_digit != True]
print("digit_removal: ", digit_removal, "\n")

freq = Counter(digit_removal)
common_words = freq.most_common(5)
print("freq:", common_words, "\n")

#print([(X, X.ent_iob_, X.ent_type_) for X in doc], "\n")  # print the entity type of each word
labels = [x.label_ for x in doc.ents]
#print(Counter(labels), "\n")  # print the number of word that has "CARDINAL" entity

#for token in doc:
# print(token.text, "-->", token.dep_)  # print the dependency of each word

# ------------------------------------------------------------------------------------------
