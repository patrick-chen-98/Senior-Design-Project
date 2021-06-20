import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import json


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def create_training_data(file, type):
    file = open(file, "r")
    data = file.readlines()
    train = []
    for k in data:
        train.append(k.rstrip())
    # print(data)
    patterns = []
    for item in train:
        pattern = {
            "label": type,
            "pattern": item
        }
        patterns.append(pattern)
    return patterns


def generate_rules(patterns):
    nlp = English()
    ruler = EntityRuler(nlp)
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)
    nlp.to_disk("hp_ner")


def test_model(model, text):
    doc = nlp(text)
    results = []
    entities = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))
    if len(entities) > 0:
        results = [text, {"entities": entities}]
        print(results)
        return (results)


patterns = create_training_data("key.txt", "keyword")
generate_rules(patterns)
nlp = spacy.load("hp_ner")
TRAIN_DATA = []
# read input file, and split into sentences, and send to create training data
sentence = open("datapath.txt", "r")
a = ""
for i in sentence:
    b = i.rstrip()
    a += b
# print(a)
a = a.split(".")
# print(a)
i = 0
for abs in a:
    abs = abs.strip()
    results = test_model(nlp, abs)
    if results != None:
        TRAIN_DATA.append(results)

save_data("train_data.json", TRAIN_DATA)
