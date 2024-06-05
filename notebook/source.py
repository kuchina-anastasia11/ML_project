from gensim.models import KeyedVectors
import numpy as np
import pickle
class model():
    def __init__(self,):
        self.wv = KeyedVectors.load("../models/word2vec.kvmodel", mmap='r')
        self.clf1 =  pickle.load(open('../models/clf1.sav', 'rb'))
        self.clf2 =  pickle.load(open('../models/clf2.sav', 'rb'))
    def vectorize(self, sentence):
        words = sentence.split()
        words_vecs = [self.wv[word] for word in words if word in self.wv]
        if len(words_vecs) == 0:
            return np.zeros(100)
        words_vecs = np.array([words_vecs])
        return words_vecs.mean(axis=0)
    def predict(self, sentence: str):
        inp = np.array(self.vectorize(sentence))
        pred1 = self.clf1.predict(inp)[0]
        print(pred1)
        if pred1 == 1:
            return 'не токсично'
        else:
            pred2 = self.clf2.predict(inp)[0]
            print(pred2,'f')
            if pred2==1:
                return 'токсично'
            if pred2==2:
                return 'непристойный'
            if pred2==3:
                return 'угроза'
            if pred2==4:
                return 'оскорбление'
            if pred2==5:
                return 'ненависть к идентичности'
