#   started 03/31/2021
#   goal create tic tac toe game and have agents play it

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
''' |0|0|0|
    |0|0|0|     How the board starts
    |0|0|0|'''

class player:
    def __init__(self, sign=1):
        self.sign = sign

        self.model = Sequential()
        self.model.add(Dense(9, activation='relu', input_shape=(9,9)))
        self.model.add(Flatten())
        self.model.add(Dense(9, activation='relu'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(optimizer='adam')

class board_env:
    def __init__(self):
        self.board = np.array([
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ])

p1 = player()
print(p1.sign)
b1 = board_env()
print(b1.board)