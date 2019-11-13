import sys
import numpy as np

print('Number of arguments:', len(sys.argv), 'arguments.')
print('types:', type(sys.argv[0]), type(sys.argv[1]))
print('ciao bello ti stampo un numero')
print(np.random.randn(int(sys.argv[1])))
