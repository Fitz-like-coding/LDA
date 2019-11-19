'''mixture of unigram model'''

import numpy as np 
from numpy.random import dirichlet
from numpy.random import multinomial


K = 2
V = 8
# prior
Phi = dirichlet([1]*V, K)
Theta = dirichlet([1]*K)
print (Phi)
print (Theta)

z =  multinomial(1, doc_topic_dis)
print (z)

word = multinomial(1, topic_word_dis[0])
print (word)