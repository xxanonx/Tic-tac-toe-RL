#   started 03/31/2021
#   goal create tic tac toe game and have agents play it

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


class player:
    def __init__(self, sign=1):
        self.sign = sign
        self.score = 0
        self.opponent_score = 0


        #Player Model
        #Looks at state and decides what move to make
        self.Actor_model = Sequential()
        self.Actor_model.add(Dense(9, activation='relu', input_shape=(3, 3)))
        self.Actor_model.add(Flatten())
        self.Actor_model.add(Dense(9, activation='relu'))
        self.Actor_model.add(Dense(2, activation='linear'))
        self.Actor_model.compile(optimizer='adam')
        self.Actor_model.summary()

        #Critic Model
        # Looks at state and thinks its a good state for actor or not
        self.Critic_model = Sequential()
        self.Critic_model.add(Dense(9, activation='relu', input_shape=(3, 3)))
        self.Critic_model.add(Flatten())
        self.Critic_model.add(Dense(4, activation='relu'))
        self.Critic_model.add(Dense(2, activation='linear'))
        self.Critic_model.compile(optimizer='adam')
        self.Critic_model.summary()

class board_env:
    def __init__(self):
        self.p1 = player(1)
        print(self.p1.sign)
        self.game_over = False
        self.whose_turn = False
        self.reset()

    def reset(self):
        ''' |0|0|0|
            |0|0|0|     How the board starts
            |0|0|0|'''
        self.board = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        self.p1.score = 0
        self.p1.opponent_score = 0
        self.game_over = False
        self.whose_turn = False

    def look_for_win(self):
        # if no zeros are found the game is over, this can tell if its a tie
        board_filled = self.board.all()
        # Checking horizontal
        for row in self.board:
            if row.all():
                # if line doesn't have zeros
                for val in row:
                    # double check that all values are the same
                    if val != row[0]:
                        break
                # player row[0] won
                self.game_over = True
                return True, row[0]
                break

        # Checking vertical
        for row in self.board.transpose():
            if row.all():
                # if line doesn't have zeros
                for val in row:
                    # double check that all values are the same
                    if val != row[0]:
                        break
                self.game_over = True
                return True, row[0]
                break

        # Checking Diagonal
        # diag_win = False
        ULC_2_LRC = self.board.diagonal()
        if ULC_2_LRC.all():
            # if line doesn't have zeros
            for val in ULC_2_LRC:
                # double check that all values are the same
                if val != ULC_2_LRC[0]:
                    break
            # diag_win = True
            self.game_over = True
            return True, ULC_2_LRC[0]

        LLC_2_URC = np.flipud(self.board).diagonal()
        if LLC_2_URC.all():
            # if line doesn't have zeros
            for val in LLC_2_URC:
                # double check that all values are the same
                if val != LLC_2_URC[0]:
                    break
            # diag_win = True
            self.game_over = True
            return True, ULC_2_LRC[0]

        if board_filled and not self.game_over:
            self.game_over = True
            return False, 0

b1 = board_env()
print(b1.board)