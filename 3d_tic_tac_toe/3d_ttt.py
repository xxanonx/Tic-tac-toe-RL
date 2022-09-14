# created by McKinley James Pollock on 9/10/22
# made for research purposes and fun

import numpy as np
"""import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Flatten"""
import random, time
import matplotlib.pyplot as plt

"""config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)"""

EMPTY_BOARD = np.zeros((3, 3, 3), dtype='int8')
ONES = np.ones((3, 3, 3))
NEG_ONES = (np.ones((3, 3, 3)) * -1)


class BoardEnv:
    def __init__(self):
        self.games_played = -1
        self.board = np.copy(EMPTY_BOARD)
        self.previous_board_O = np.copy(self.board)
        self.previous_board_X = np.copy(self.board)
        self.current_board = np.copy(self.board)
        self.game_over = False
        self.whose_turn = True
        # -1 = O and 1 = X
        # false = 0 and true = X

        self.reset()

    def reset(self):
        """[[[0. 0. 0.]
            [0. 0. 0.]
            [0. 0. 0.]]

            [[0. 0. 0.]
            [0. 0. 0.]            How the board starts
            [0. 0. 0.]]

            [[0. 0. 0.]
            [0. 0. 0.]
            [0. 0. 0.]]]"""
        self.board = np.copy(EMPTY_BOARD)
        self.previous_board_O = np.copy(self.board)
        self.previous_board_X = np.copy(self.board)
        self.current_board = np.copy(self.board)

        self.game_over = False
        self.games_played += 1
        # self.whose_turn = not self.whose_turn

    def look_for_win(self):
        # if no zeros are found the game is over, this can help tell if it's a tie
        board_filled = self.board.all()
        # BIG job of converting from 2d to 3d /
        # got to look at it from bottom to top, side to side, and other side to side

        # checking layers
        for orientation in [self.board,
                      np.moveaxis(self.board, 0, -1),
                      np.moveaxis(self.board, -1, 0)]:
            for layer in orientation:
                if self.game_over:
                    break
                if layer.any():
                    self.game_over, winner = check_layer_for_win(layer)
        if not self.game_over:
            # special diagonals
            for diag in [self.board.diagonal(), np.fliplr(self.board).diagonal()]:
                if not self.game_over:
                    if diag.any():
                        self.game_over, winner = check_layer_for_win(layer)

        if board_filled and not self.game_over:
            self.game_over = True
            winner = 0
        return self.game_over, winner

    def make_move(self, x, y, just_data=False):
        move_made = False
        self.current_board = np.copy(self.board)
        for layer in self.board:
            if move_made:
                break
            if layer[y][x] == 0:

                if not just_data:
                    move2 = np.copy(EMPTY_BOARD)
                    move2[y][x] = 1

                # back to reality
                if not just_data:
                    if self.whose_turn:
                        self.board[y][x] = 1
                        self.previous_board_X = np.copy(self.board)
                    else:
                        self.board[y][x] = -1
                        self.previous_board_O = np.copy(self.board)
                return True
        else:
            return False

    def get_state(self, human=False, random_play=False, verbose=False):
        # Whose turn matters and human matters
        if human:
            if self.whose_turn:
                piece = "X"
                # op_piece = "O"
            else:
                piece = "O"
                # op_piece = "X"

            print(f"\nYour piece is {piece}, make your move for x,y (-1, 0, 1)")
            for row in self.board:
                line = "|"
                for val in row:
                    if val == -1:
                        line += " O |"
                    elif val == 1:
                        line += " X |"
                    else:
                        line += "   |"
                print(line)
            while True:
                move = input("what is your move? ")
                move1 = move.split(",")
                move2 = move.split(",")
                if ("," in move) and (5 >= len(move) >= 3) and ((move1[0].isdigit() or move1[0] == "-1") and (move1[1].isdigit() or move1[1] == "-1")):
                    x = -1
                    # print(move1)
                    for axis in move1:
                        x += 1
                        # print(axis)
                        if -1 <= int(axis) <= 1:
                            if int(axis) == -1:
                                move1[x] = 0
                                continue
                            elif int(axis) == 0:
                                move1[x] = 1
                                continue
                            elif int(axis) == 1:
                                move1[x] = 2
                                continue
                    # print(move1)
                    if (move1[0] != move2[0]) and (move1[1] != move2[1]):
                        if self.make_move(move1[0], move1[1]):
                           # print(f"{move} is acceptable")
                            break

                print(f"{move} IS INVALID")

        else:
            # Computers turn
            # print("Computer's turn")
            ai_can_skip = False
            move = [0, 0]
            while True:
                self.current_board = np.copy(self.board)
                if verbose:
                    # print([self.current_board, self.previous_board])
                    if not self.whose_turn:
                        c_predict = self.p1.Critic_model(np.array([[self.current_board.astype('float32'),
                                                                            self.previous_board_O.astype('float32'),
                                                                            self.neg_ones.astype('float32')]])).numpy()
                        value_layer = (np.ones((3, 3)) * c_predict[0][0])
                        a_predict = (self.p1.Actor_model(np.array([[self.current_board.astype('float32'),
                                                                           self.previous_board_O.astype('float32'),
                                                                           self.neg_ones.astype('float32'),
                                                                           value_layer.astype('float32')]]))[0]).numpy()
                        print(f"NN Value for O's: {c_predict}")
                        print(a_predict)
                    else:
                        c_predict = self.p1.Critic_model(np.array([[self.current_board.astype('float32'),
                                                                            self.previous_board_X.astype('float32'),
                                                                            self.ones.astype('float32')]])).numpy()
                        value_layer = (np.ones((3, 3)) * c_predict[0][0])
                        a_predict = (self.p1.Actor_model(np.array([[self.current_board.astype('float32'),
                                                                           self.previous_board_X.astype('float32'),
                                                                           self.ones.astype('float32'),
                                                                           value_layer.astype('float32')]]))[0]).numpy()

                        print(f"NN Value for X's: {c_predict}")
                        print(a_predict)

                if random_play:
                    random.seed(time.time_ns())
                    move[0] = random.randint(-1, 1)
                    move[1] = random.randint(-1, 1)
                else:
                    if ai_can_skip:
                        pred[move[1]][move[0]] = 0
                    elif not self.whose_turn:
                        c_predict = self.p1.Critic_model(np.array([[self.current_board.astype('float32'),
                                                                            self.previous_board_O.astype('float32'),
                                                                            self.neg_ones.astype('float32')]])).numpy()
                        value_layer = (np.ones((3, 3)) * c_predict[0][0])
                        pred = (self.p1.Actor_model(np.array([[self.current_board.astype('float32'),
                                                                       self.previous_board_O.astype('float32'),
                                                                       self.neg_ones.astype('float32'),
                                                                       value_layer.astype('float32')]]))[0]).numpy()
                    else:
                        c_predict = self.p1.Critic_model(np.array([[self.current_board.astype('float32'),
                                                                            self.previous_board_X.astype('float32'),
                                                                            self.ones.astype('float32')]])).numpy()
                        value_layer = (np.ones((3, 3)) * c_predict[0][0])
                        pred = (self.p1.Actor_model(np.array([[self.current_board.astype('float32'),
                                                                       self.previous_board_X.astype('float32'),
                                                                       self.ones.astype('float32'),
                                                                       value_layer.astype('float32')]]))[0]).numpy()
                    if verbose:
                        print('Algorithm playing!')
                        # print(pred)
                    if pred.any():
                        ai_y = 0
                        max = pred.ravel().max()
                        for row in pred:
                            row_max = int(tf.argmax(row))
                            if row[row_max] == max:
                                ai_can_skip = True
                                move = [row_max, ai_y]
                                break
                            ai_y += 1
                    else:
                        random.seed(time.time_ns())
                        move[0] = random.randint(-1, 1)
                        move[1] = random.randint(-1, 1)

                move1 = move.copy()
                if not ai_can_skip:
                    x = -1
                    for axis in move1:
                        x += 1
                        # print(axis)
                        if -1 <= int(axis) <= 1:
                            if int(axis) == -1:
                                move1[x] = 0
                                continue
                            elif int(axis) == 0:
                                move1[x] = 1
                                continue
                            elif int(axis) == 1:
                                move1[x] = 2
                                continue
                    # print(move1)
                if ((move1[0] != move[0]) and (move1[1] != move[1])) or ai_can_skip:
                    if self.make_move(move1[0], move1[1]):
                        if not random_play and ai_can_skip and verbose:
                            print(f"{move} is acceptable")
                        break

        done, winner_ = self.look_for_win()
        if done:
            self.make_move(0, 0, just_data=True)
            value_board = self.value_of_board(done, True, symbol=winner_)
            if self.p1.sign == winner_:
                self.p1.score += 1
            else:
                self.p1.opponent_score += 1
            if winner_ == -1 and value_board > 0.7:
                self.actor_training.extend(self.round_buffer_O)
                self.actor_moves.extend(self.move_buffer_O)
                self.actor_training.append(self.round_buffer_X[-1])
                self.actor_moves.append(self.move_buffer_O[-1])
            elif winner_ == 1 and value_board > 0.7:
                self.actor_training.extend(self.round_buffer_X)
                self.actor_moves.extend(self.move_buffer_X)
                self.actor_training.append(self.round_buffer_O[-1])
                self.actor_moves.append(self.move_buffer_X[-1])
            self.round_buffer_O.clear()
            self.move_buffer_O.clear()
            self.round_buffer_X.clear()
            self.move_buffer_X.clear()
            if verbose:
                print(self.board)
                if winner_ == -1:
                    print("O's WON!!!")
                elif winner_ == 1:
                    print("X's WON!!!!")
                else:
                    print("DRAW")

            self.reset()

        # end of turn
        self.whose_turn = not self.whose_turn


def check_layer_for_win(layer):
    game_over = False
    winner = 0
    # Checking horizontal and vertical
    for board in [layer, layer.transpose()]:
        for row in board:
            if not game_over:
                in_a_row = 0
                # print(row)
                if row.all():
                    # if line doesn't have zeros
                    for valr in row:
                        # double check that all values are the same
                        if int(valr) != int(row[0]):
                            break
                        else:
                            in_a_row += 1
                    if in_a_row == 3:
                        # player row[0] won
                        game_over = True
                        winner = row[0]
                        break

    # Checking Diagonal
    # diag_win = False

            break

        for diag in [board.diagonal(), np.flipud(board).diagonal()]:
            if diag.all():
                in_a_row = 0
                # if line doesn't have zeros
                for vald1 in diag:
                    # double check that all values are the same
                    if int(vald1) != int(diag[0]):
                        break
                    else:
                        in_a_row += 1
                if in_a_row == 3:
                    game_over = True
                    winner = diag[0]

    return game_over, winner
