import spacy
from flair.data import Sentence
from flair.models import SequenceTagger

# Load models
nlp = spacy.load("en_core_web_sm")
flair_tagger = SequenceTagger.load("flair/ner-english-large")

def extract_named_entities(text):
    """Extract named entities using Hugging Face's flair model."""
    sentence = Sentence(text)
    flair_tagger.predict(sentence)
    return [(entity.text, entity.get_label("ner").value) for entity in sentence.get_spans("ner")]

def extract_activities(text):
    """Extract activities (verbs + objects) using spaCy."""
    doc = nlp(text)
    activities = []
    for token in doc:
        if token.pos_ == "VERB":  # Find verbs
            obj = [child.text for child in token.children if child.dep_ in ["dobj", "prep", "pobj"]]
            activity_phrase = token.text + " " + " ".join(obj) if obj else token.text
            activities.append(activity_phrase)
    return activities