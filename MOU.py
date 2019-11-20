'''
mixture of unigram model
'''


import numpy as np 
from scipy.stats import dirichlet

class MOU(object):
    def __init__(self, n_components, max_iter, Alpha, Beta, seed=None):
    
        self.K = n_components
        self.max_iter = max_iter
        self.seed = seed
        self.Alpha = Alpha
        self.Beta = Beta
        self.Phi = None
        self.Theta = None
        self.Z = None

    def _sampler(self, doc_term):
        docs_number = doc_term.shape[0]
        vobs_number = doc_term.shape[1]
        Phi = dirichlet.rvs([self.Beta]*vobs_number, size=self.K, random_state=self.seed)
        Theta = dirichlet.rvs([self.Alpha]*self.K, random_state=self.seed)
        Z = [[self.Alpha]*self.K] * doc_term.shape[0] * Theta

        itr = 0
        while itr <= self.max_iter:
            # update Phi and Beta
            e = self.Beta + np.multiply(doc_term, Z.T.reshape(self.K, doc_term.shape[0],1)).sum(axis=1) - 1 + 1e-20
            for k in range(len(e)):
                Phi[k] = dirichlet.rvs(e[k], random_state=None)

            # update Z
            temp = np.power(np.repeat(Phi.reshape(Phi.shape[0], -1, Phi.shape[1]), doc_term.shape[0], axis=1), doc_term)
            temp = np.prod(temp, axis=2)
            Z = np.multiply(temp, Z.T).T
            Z = Z/Z.sum(axis=1, keepdims=True)

            # update Theta
            d = self.Alpha + Z.sum(axis=0, keepdims=True)
            Theta = dirichlet.rvs(d[0], random_state=None)

            itr += 1

        return Phi, Theta, Z

    def fit(self, doc_term):
        self.Phi, self.Theta, self.Z = self._sampler(doc_term)

if __name__ == '__main__':
    vocabulary = ['literary', 'literature', 'authors', 'century', 'texts', 'writers', 'economic', 'critique']
    doc_term = np.array([[0,0,0,4,0,0,2,5],
                        [0,0,0,0,0,0,6,11],
                        [0,0,0,3,0,0,8,0],
                        [0,0,0,2,1,0,6,16],
                        [0,1,0,5,1,0,3,13],
                        [0,0,0,0,0,0,5,6],
                        [10,3,0,4,0,1,0,0],
                        [13,1,7,0,0,5,0,0],
                        [7,3,0,4,1,8,0,0],
                        [20,14,3,0,0,0,0,0],
                        [5,6,5,0,0,10,0,0],
                        [9,7,0,2,0,1,0,0],
                        [3,5,3,0,0,6,0,0],
                        [8,13,3,1,1,3,0,0],
                        [9,3,4,0,0,6,0,0],
                        [11,7,4,0,1,6,0,0],
                        [2,3,0,1,1,1,0,0],
                        [5,2,13,0,0,5,0,0],
                        [7,3,6,1,0,11,0,0],
                        [5,9,8,2,0,4,0,0]])

    mou = MOU(n_components=2, max_iter=600, Alpha=1, Beta=1, seed=1)
    mou.fit(doc_term)

    #topic words distribution
    for k, p in enumerate(mou.Phi):
        print ('topic ',k)
        for w in np.argsort(p)[::-1][:5]:
            print (vocabulary[w], p[w])
        print ()
            
    #doc topic distribution
    print ('Docs Topic0 Topic1' )
    for i in range(len(mou.Z)):
        print (i, mou.Z[i][0], mou.Z[i][1])
