"""Identical topic preprocessing during fitting and inference."""
from functools import lru_cache


@lru_cache(maxsize=1)
def topic_parser():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def topic_input(texts):
    return [" ".join(t.lemma_.lower() for t in doc if t.pos_ in {"NOUN", "VERB", "ADJ"} and not t.is_stop)
            for doc in topic_parser().pipe(texts, batch_size=64)]
