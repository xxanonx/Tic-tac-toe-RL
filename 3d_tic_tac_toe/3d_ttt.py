# created by McKinley James Pollock on 9/10/22
# made for research purposes and fun

import numpy as np
from numba import jit
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

random.seed(time.time_ns())


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

    def vis(self):
        visualize3d(self.board)

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
        layer_num = 0
        for layer in self.board:
            if move_made:
                break
            if layer[y][x] == 0:
                if not just_data:
                    move2 = np.copy(EMPTY_BOARD)
                    move2[y][x] = 1
                    if self.whose_turn:
                        self.board[layer_num][y][x] = 1
                        self.previous_board_X = np.copy(self.board)
                        move_made = True
                    else:
                        self.board[layer_num][y][x] = -1
                        self.previous_board_O = np.copy(self.board)
                        move_made = True
            layer_num += 1
        # return false if the move can't be made
        return move_made

    def get_state(self, human=False, random_play=False, verbose=False):
        # Whose turn matters and human matters
        input_max = (IN_A_ROW - 1)
        while True:
            final_move = [-20, -20]
            if human:
                if self.whose_turn:
                    piece = "X/red"
                    # op_piece = "O"
                else:
                    piece = "O/blue"
                    # op_piece = "X"

                print(f"\nYour piece is {piece}, make your move for x,y (0, 1, 2...)")

                if verbose:
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
                        num = 0
                        for axis_of_input in move.split(","):
                            # hopefully only two digits
                            try:
                                final_move[num] = int(axis_of_input)
                            except:
                                pass
                            else:
                                if 0 <= final_move[num] <= input_max:
                                    num += 1

                        if (0 <= final_move[0] <= input_max) and (0 <= final_move[1] <= input_max):
                            break

                    print(f"{move} IS INVALID")

            else:
                if random_play:
                    final_move[0] = random.randint(0, input_max)
                    final_move[1] = random.randint(0, input_max)

            if self.make_move(final_move[0], final_move[1]):
                if verbose:
                    print(f"{final_move} is acceptable")
                break
            else:
                print(f"it seems that {final_move} is not legal")

        done, winner_ = self.look_for_win()
        if done:
            if verbose:
                print(self.board)
                if winner_ == -1:
                    print("O's WON!!!")
                elif winner_ == 1:
                    print("X's WON!!!!")
                else:
                    print("DRAW")

            # self.reset()

        # end of turn
        self.whose_turn = not self.whose_turn


# for 2D

def check_layer_for_win(layer2d: np.ndarray):
    game_over = False
    winner = 0
    # Checking horizontal and vertical
    for board2d in [layer2d, np.copy(layer2d.transpose())]:
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
                    '''print(row)
                    print('player won')'''
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
            x_coord = 0
            for row in layer:
                y_coord = 0
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


B1 = BoardEnv()

while B1.games_played <= 10:
    start = time.perf_counter()
    B1.get_state(False, True)
    if B1.game_over:
        end = time.perf_counter()
        print(str((end - start)*1000) + " milliseconds")
        B1.vis()
        B1.reset()