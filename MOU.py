'''mixture of unigram model'''

import numpy as np 
from numpy.random import dirichlet
from numpy.random import multinomial

topic_word_dis = dirichlet([1]*8, 2)
print (topic_word_dis)

doc_topic_dis = dirichlet([1]*2)
print (doc_topic_dis)

z =  multinomial(1, doc_topic_dis)
print (z)

word = multinomial(1, topic_word_dis[0])
print (word)