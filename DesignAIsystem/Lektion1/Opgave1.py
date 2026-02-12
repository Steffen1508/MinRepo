import re
import math
from sklearn.datasets import fetch_20newsgroups


categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset="train",categories=categories, shuffle=True,random_state=42,remove=("headers","footers","quotes"))

n_docs = twenty_train.data[0:10]


def tokenize_doc(text):
    lc = text.lower()
    clean = re.sub(r"[^\w\s]","",lc)
    lw = re.findall(r"\b\w+\b",clean)
    token = lw

    return token

tokenized_docs = []

for i in n_docs:
    sub = tokenize_doc(i)
    tokenized_docs.append(sub)



def count_words(doc_tokens):
    counts = {}
    for i in doc_tokens:
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1
    tf = {}
    total_words = len(doc_tokens)
    for i in counts:
        tf[i] = counts[i]/total_words
    return tf


doc_tokens = tokenized_docs[0]
tf_values = count_words(doc_tokens)

def compute_df(tokenized_docs):
    df = {}
    for doc in tokenized_docs:
        unique_words = set(doc)
        for word in unique_words:
            if word in df:
                df[word] += 1
            else:
                df[word] = 1
    return df

def compute_idf(df,tokenized_docs):
    N = len(tokenized_docs)
    idf = {}
    for word in df:
        temp = math.log(N/df[word])
        idf[word] = temp
    return idf


def compute_tfidf(tf,idf):
    tfidf = {}
    for word in tf:
        tfidf[word] = tf[word] * idf[word]

    return tfidf


df = compute_df(tokenized_docs)
idf = compute_idf(df, tokenized_docs)


tfidf = compute_tfidf(tf_values, idf)


sorted_list = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)


print(sorted_list[:10])