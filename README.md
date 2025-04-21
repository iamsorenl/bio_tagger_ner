# bio_tagger_ner
A BIO tagger for four kinds of entities: people, locations, organizations, and names of other miscellaneous entities.

Files:

**conlleval.py**: This is a Python script for evaluating NER performance, possibly calculating precision, recall, and F1-score based on the CoNLL 2003 shared task format.

Files in data folder:

**gazetteer.txt**: A list of locations (LOC) that can be used as a feature in your NER system to help identify place names.

**ner.dev**: The development dataset for your NER task, formatted in the CoNLL 2003 style (tokens with POS tags and named entity labels).

**ner.test**: The test dataset for evaluating your trained NER model.

**ner.train**: The training dataset used to train your NER model.
