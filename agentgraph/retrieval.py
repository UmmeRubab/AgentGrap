import re
from collections import Counter

STOPWORDS = set("""
a an the and or of to in on for with what which who where
when did does is was were are by from at as how that this
""".split())


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOPWORDS]


def flatten_context(context):
    sentences = []

    for document in context:
        for index, sentence in enumerate(
            document["sentences"]
        ):
            sentences.append({
                "title": document["title"],
                "index": index,
                "text": sentence,
            })

    return sentences


def retrieve(question, context, top_k=6):
    query = set(tokenize(question))
    candidates = flatten_context(context)

    def score(candidate):
        words = Counter(
            tokenize(
                candidate["title"] + " " + candidate["text"]
            )
        )

        return sum(
            min(words.get(word, 0), 3)
            for word in query
        )

    return sorted(
        candidates,
        key=lambda item: -score(item),
    )[:top_k]
