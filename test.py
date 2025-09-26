print('this is initial commit')

import numpy as np

S=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(S.reshape(3,2,2))
print(S.reshape(2,3,2))