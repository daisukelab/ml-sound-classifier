import sys
sys.path.append('..')
from lib_train import *

X = [
    [1.0,0.0,0.0],
    [0.8,0.0,0.1],
    [0.5,0.5,0.0],
    [0.4,0.4,0.0],
    [0.6,0.7,0.0],
    [0.1,0.5,0.5],
]
y = [
    'kylin',
    'kylin',
    'tiger',
    'tiger',
    'tiger',
    'eleph',
]

# balance_class_by_over_sampling, balance_class_by_under_sampling
correct = (np.array([[1. , 0. , 0. ],
       [0.8, 0. , 0.1],
       [0.5, 0.5, 0. ],
       [0.4, 0.4, 0. ],
       [0.6, 0.7, 0. ],
       [0.1, 0.5, 0.5],
       [0.1, 0.5, 0.5],
       [0.1, 0.5, 0.5],
       [1. , 0. , 0. ]]), np.array(['kylin', 'kylin', 'tiger', 'tiger', 'tiger', 'eleph', 'eleph',
       'eleph', 'kylin']))
print('balance_class_by_over_sampling?',
    test_equal(correct, balance_class_by_over_sampling(X, y)))
correct = (np.array([[0.1, 0.5, 0.5],
        [0.8, 0. , 0.1],
        [0.4, 0.4, 0. ]]), np.array(['eleph', 'kylin', 'tiger']))
print('balance_class_by_under_sampling?',
    test_equal(correct, balance_class_by_under_sampling(X, y)))