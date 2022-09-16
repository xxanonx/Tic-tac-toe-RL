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

IN_A_ROW = 3
EMPTY_BOARD = np.zeros((IN_A_ROW, IN_A_ROW, IN_A_ROW), dtype='int8')
ONES = np.ones((IN_A_ROW, IN_A_ROW, IN_A_ROW), dtype='int8')
NEG_ONES = (np.ones((IN_A_ROW, IN_A_ROW, IN_A_ROW), dtype='int8') * -1)


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
            [0. 0. 0.]            How the board starts in 3x3x3 config
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
        winner = 0
        # look at board from bottom to top, side to side, and other side to side

        # checking layers
        for orientation in [self.board,
                            np.moveaxis(self.board, 0, -1),
                            np.moveaxis(self.board, -1, 0)]:
            for board_layer in orientation:
                if self.game_over:
                    break
                if board_layer.any():
                    self.game_over, winner = check_layer_for_win(board_layer)
        if not self.game_over:
            # special diagonals
            # self.board.diagonal() comes out as 3x3 so already single layer
            for diag in [self.board.diagonal(), np.fliplr(self.board).diagonal()]:
                if not self.game_over:
                    if diag.any():
                        self.game_over, winner = check_layer_for_win(diag)

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
                if not just_data:
                    if self.whose_turn:
                        self.board[y][x] = 1
                        self.previous_board_X = np.copy(self.board)
                        move_made = True
                    else:
                        self.board[y][x] = -1
                        self.previous_board_O = np.copy(self.board)
                        move_made = True
        return move_made

    def get_state(self, human=False, random_play=False, verbose=False):
        # Whose turn matters and human matters
        if human:
            if self.whose_turn:
                piece = "X/red"
                # op_piece = "O"
            else:
                piece = "O/blue"
                # op_piece = "X"

            print(f"\nYour piece is {piece}, make your move for x,y (-1, 0, 1)")
            for layer in self.board:
                for row in layer:
                    line = "|"
                    for val in row:
                        if val == -1:
                            line += " O |"
                        elif val == 1:
                            line += " X |"
                        else:
                            line += "   |"
                    print(line)
                print('------')
            visualize3d(self.board)
            while True:
                move = input("what is your move? ")
                if move.__contains__(","):
                    xy = [-20, -20]
                    num = 0
                    for axis_of_input in move.split(","):
                        # hopefully only two digits
                        try:
                            xy[num] = int(axis_of_input)
                        except:
                            pass
                        else:
                            if -1 <= xy[num] <= 1:
                                xy[num] = (xy[num] + 1)
                                num += 1

                    if (0 <= xy[0] <= 2) and (0 <= xy[1] <= 2):
                        if self.make_move(xy[0], xy[1]):
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


# for 2D
def check_layer_for_win(layer2d):
    game_over = False
    winner = 0
    # Checking horizontal and vertical
    for board2d in [layer2d, layer2d.transpose()]:
        for row in board2d:
            if game_over:
                break
            in_a_row = 0
            if row.all():
                # if line doesn't have zeros
                for valr in row:
                    # double check that all values are the same
                    if int(valr) != int(row[0]):
                        break
                    else:
                        in_a_row += 1
                if in_a_row == IN_A_ROW:
                    print(row)
                    print('player won')
                    game_over = True
                    winner = row[0]
                    break

    '''[[0. 0. 0.]
        [0. 0. 0.]      Board/layer
        [0. 0. 0.]]'''
    # Checking Diagonal
    for diag in [layer2d.diagonal(), np.flipud(layer2d).diagonal()]:
        if game_over:
            break
        # if line doesn't have zeros
        if diag.all():
            in_a_row = 0
            for vald1 in diag:
                # double check that all values are the same
                if int(vald1) != int(diag[0]):
                    break
                else:
                    in_a_row += 1
            if in_a_row == IN_A_ROW:
                print(row)
                print('player won')
                game_over = True
                winner = diag[0]

    return game_over, winner


# for 3d
def visualize3d(board3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for color, add in [('b', -1),
                       ('r', 1),
                       ('0.8', 0)]:
        x = []
        y = []
        z = []
        layer_num = 0
        temp_board = np.copy(board3d)
        for layer in temp_board:
            x_coord = -1
            for row in layer:
                y_coord = -1
                for item in row:
                    if item == add:
                        for i in range(20):
                            x.append(x_coord)
                            y.append(y_coord)
                            z.append(layer_num)
                            if add == 0:
                                break
                    y_coord += 1
                x_coord += 1
            layer_num += 1
        ax.scatter(x, y, z, color=color)

    plt.show()
