import sys
import gensim

print('Random word representation')
sys.stdout.flush()
path_word2vec = '/om/user/vanessad/IMDb_framework/GoogleNews-vectors-negative300.bin'
embed_dct = {'word2vec': {'dimension': 300,
                          'model': gensim.models.KeyedVectors.load_word2vec_format(path_word2vec, binary=True)},
             'glove': {'dimension': 100, 'model': None}
             }

print('Random word representation')
sys.stdout.flush()

print(embed_dct['word2vec']['model']['king'])
sys.stdout.flush()
