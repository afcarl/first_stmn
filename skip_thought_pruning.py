from gensim.models import Word2Vec
import numpy
import re

import os

import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy
import nltk

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize

import skipthoughts

#you need to find a way to iterate for each statement/question pair; look at "for line in f:" in theano.util
def prune_thoughts(dataset, questions, input_dir):
    i = open(input_dir)
    text = i.read()
    clean = re.sub("[0-9]", "", text)
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(clean)
    X = sents
    #print sents
    Y = dataset
    model = skipthoughts.load_model()
    vectors = skipthoughts.encode(model, X)
    #nearest_neighbor = skipthoughts.nn(model, X, vectors, Y, k=5)
    #print dataset
    #print questions
    return X

def prune_statements(dataset, questions):
    total_old = 0
    total_new = 0

    wvs = Word2Vec(dataset, min_count=0)

    for i in range(len(questions)):
        question = questions[i]
        new_statements = []
        old_statements = question[2][:-1]

        # Use word vectors and keep only the top 5

        sims = []
        q = question[2][-1]
        for s in old_statements:
            sims.append(wvs.n_similarity(q,s))

        sims2 = map(lambda x: x if type(x) is numpy.float64 else 0.0, sims)
        top = sorted(range(len(sims2)), key=sims2.__getitem__, reverse=True)
        new_statements = map(lambda x: old_statements[x], top[:5])

        questions[i][2] = new_statements
        total_old += len(old_statements)
        total_new += len(new_statements)
        #print("Question: ", questions[i][2][-1], " before %d after %d" % (len(old_statements), len(new_statements)))

    print("Before %d After %d" % (total_old, total_new))
    return questions