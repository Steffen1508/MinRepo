import re
import math
from sklearn.datasets import fetch_20newsgroups


# ------------------------------------------------------
# Indlæs datasæt og vælg et mindre subset (10 dokumenter)
# ------------------------------------------------------

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

# Loader træningsdata og fjerner headers/footers/quotes
twenty_train = fetch_20newsgroups(
    subset="train",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes")
)

# Vi bruger kun de første 10 dokumenter som vores korpus
n_docs = twenty_train.data[0:10]

# Liste der skal indeholde tokenized version af alle dokumenter
tokenized_docs = []


# ------------------------------------------------------
# Tokenization: gør tekst klar til beregning
# ------------------------------------------------------

def tokenize_doc(text):
    # Gør alt lowercase så "The" og "the" bliver ens
    lc = text.lower()

    # Fjern tegnsætning (behold kun bogstaver, tal og whitespace)
    clean = re.sub(r"[^\w\s]", "", lc)

    # Find alle ord (tokens)
    lw = re.findall(r"\b\w+\b", clean)

    return lw


# Tokenize alle dokumenter i korpuset
for doc in n_docs:
    tokens = tokenize_doc(doc)
    tokenized_docs.append(tokens)


# ------------------------------------------------------
# TF (Term Frequency)
# Hvor ofte optræder et ord i ét dokument?
# TF = (antal gange ordet optræder) / (antal ord i dokumentet)
# ------------------------------------------------------

def count_words(doc_tokens):
    counts = {}

    # Tæl hvor mange gange hvert ord optræder
    for word in doc_tokens:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    # Konverter counts til TF-værdier
    tf = {}
    total_words = len(doc_tokens)

    for word in counts:
        tf[word] = counts[word] / total_words

    return tf


# ------------------------------------------------------
# DF (Document Frequency)
# Hvor mange dokumenter indeholder ordet mindst én gang?
# ------------------------------------------------------

def compute_df(tokenized_docs):
    df = {}

    for doc in tokenized_docs:
        # Brug set så hvert ord kun tælles én gang pr dokument
        unique_words = set(doc)

        for word in unique_words:
            if word in df:
                df[word] += 1
            else:
                df[word] = 1

    return df


# ------------------------------------------------------
# IDF (Inverse Document Frequency)
# IDF = log(N / DF)
# N = antal dokumenter i korpus
# ------------------------------------------------------

def compute_idf(df, tokenized_docs):
    N = len(tokenized_docs)
    idf = {}

    for word in df:
        idf[word] = math.log(N / df[word])

    return idf


# ------------------------------------------------------
# TF-IDF
# TF-IDF = TF * IDF
# Beregnes kun for ordene i target dokumentet
# ------------------------------------------------------

def compute_tfidf(tf, idf):
    tfidf = {}

    for word in tf:
        tfidf[word] = tf[word] * idf[word]

    return tfidf


# ------------------------------------------------------
# Beregninger
# ------------------------------------------------------

# Vælg første dokument som target
doc_tokens = tokenized_docs[0]

# Beregn TF for target dokument
tf_values = count_words(doc_tokens)

# Beregn DF for hele korpuset
df = compute_df(tokenized_docs)

# Beregn IDF
idf = compute_idf(df, tokenized_docs)

# Beregn TF-IDF for target dokument
tfidf = compute_tfidf(tf_values, idf)


# ------------------------------------------------------
# Sortér og print resultat
# ------------------------------------------------------

# Sortér TF-IDF scores (højeste først)
sorted_items = sorted(tfidf.items(),
                      key=lambda item: item[1],
                      reverse=True)

print("\n===== TF-IDF RESULTAT =====")
print(f"Antal dokumenter i korpus: {len(tokenized_docs)}")
print(f"Antal ord i target dokument: {len(doc_tokens)}")

print("\nFørste 200 tegn af target dokument:")
print("-" * 50)
print(n_docs[0][:200])
print("-" * 50)

print("\nTop 10 ord med højest TF-IDF:\n")

for word, score in sorted_items[:10]:
    print(f"{word:<15} {score:.6f}")
