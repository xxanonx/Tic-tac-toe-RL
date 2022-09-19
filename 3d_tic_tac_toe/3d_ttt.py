# created by McKinley James Pollock on 9/10/22
# made for research purposes and fun

import numpy as np
from numba import jit
from tqdm import tqdm
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
games_played = 0
actual_play = False


class BoardEnv:

    def __init__(self):
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
        global games_played
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
        if actual_play:
            games_played += 1
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
                for board in [board_layer, board_layer.transpose()]:
                    if self.game_over:
                        break
                    if board_layer.any():
                        self.game_over, winner = check_layer_for_win(board)
        if not self.game_over:
            # special diagonals
            # self.board.diagonal() comes out as 3x3 so already single layer
            for diag in [self.board.diagonal(), np.fliplr(self.board).diagonal()]:
                for diag_board in [diag, diag.transpose()]:
                    if not self.game_over:
                        if diag_board.any():
                            self.game_over, winner = check_layer_for_win(diag_board)

        if board_filled and not self.game_over:
            self.game_over = True
            winner = 0
        return self.game_over, winner

    def make_move(self, x: int, y: int, just_data=False):
        move_made: bool = False
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
                if verbose:
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


class Player:
    def __init__(self, sign=1):
        self.sign = sign
        self.score = 0
        self.opponent_score = 0
        self.Critic_model = Sequential()
        self.Actor_model = Sequential()
        self.do_not_init_a = False
        self.do_not_init_c = False

    def init_critic(self):
        # Critic Model
        # Looks at state and thinks its a good state for actor or not
        self.Critic_model.add(Dense(9, input_shape=(3, 3, 3), activation='relu'))
        self.Critic_model.add(Flatten())
        self.Critic_model.add(Dense(9, activation='relu'))
        self.Critic_model.add(Dense(9, activation='relu'))
        self.Critic_model.add(Dense(1, activation='tanh'))
        self.Critic_model.compile(
            optimizer='adam',
            loss='huber_loss',
            metrics=['accuracy']
        )
        self.Critic_model.summary()

    def init_actor(self):
        # Actor trained on wins
        self.Actor_model.add(Dense(18, input_shape=(4, 3, 3), activation='relu'))
        self.Actor_model.add(Dense(18, activation='relu'))
        self.Actor_model.add(Flatten())
        self.Actor_model.add(Dense(27, activation='relu'))
        self.Actor_model.add(Dense(9, activation='sigmoid'))
        self.Actor_model.add(Reshape((3, 3)))
        self.Actor_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.Actor_model.summary()

    def teach_critc(self, state, value):
        if not self.do_not_init_c:
            self.init_critic()
            self.do_not_init_c = True
        div = int((len(state) * 0.7))
        train_x = np.copy(state[:div]).astype('float32')
        validation_x = np.copy(state[div:]).astype('float32')
        train_y = np.copy(value[:div]).astype('float32')
        validation_y = np.copy(value[div:]).astype('float32')

        print(train_x.dtype)
        print(train_y.dtype)
        print(train_x.shape)
        print(train_y.shape)

        print(train_x)
        print(train_y)

        self.Critic_model.fit(train_x, train_y,  validation_data=(validation_x, validation_y), epochs=10)
        self.Critic_model.save('/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/2D_tic_tac_toe/save_models/TTT2D_Critic')

    def teach_actor(self, state, move):
        if not self.do_not_init_a:
            self.init_actor()
            self.do_not_init_a = True
        div = int((len(state) * 0.7))
        train_x = np.copy(state[:div]).astype('float32')
        validation_x = np.copy(state[div:]).astype('float32')
        train_y = np.copy(move[:div]).astype('float32')
        validation_y = np.copy(move[div:]).astype('float32')

        print(train_x.dtype)
        print(train_y.dtype)
        print(train_x.shape)
        print(train_y.shape)

        print(train_x)
        print(train_y)

        self.Actor_model.fit(train_x, train_y, validation_data=(validation_x, validation_y), epochs=20)
        self.Actor_model.save('/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/2D_tic_tac_toe/save_models/TTT2D_actor')

    def load_models(self):
        self.Critic_model = load_model(
            '/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/2D_tic_tac_toe/save_models/TTT2D_Critic')
        # self.Critic_model.summary()
        self.do_not_init_c = True
        self.Actor_model = load_model(
            '/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/2D_tic_tac_toe/save_models/TTT2D_actor')
        self.do_not_init_a = True


# for 2D
def check_layer_for_win(layer2d: np.array = EMPTY_BOARD[0]):
    game_over = False
    winner = 0
    # Checking horizontal and vertical
    # for board2d in [layer2d, np.copy(layer2d.transpose())]:
    for row in layer2d:
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
    diag = layer2d.diagonal()
    if not game_over:
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
                '''print(row)
                print('player won')'''
                game_over = True
                winner = diag[0]
    diag2 = np.flipud(layer2d).diagonal()
    if not game_over:
        # if line doesn't have zeros
        if diag2.all():
            in_a_row = 0
            for vald1 in diag2:
                # double check that all values are the same
                if int(vald1) != int(diag2[0]):
                    break
                else:
                    in_a_row += 1
            if in_a_row == IN_A_ROW:
                '''print(row)
                print('player won')'''
                game_over = True
                winner = diag2[0]

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


list_of_boards = []
for i in tqdm(range(500)):
    list_of_boards.append(BoardEnv())

avg = 0
actual_play = True
total = time.perf_counter()
while games_played <= 1900:
    for board in tqdm(list_of_boards):
        start = time.perf_counter()
        board.get_state(False, True)
        if board.game_over:
            end = time.perf_counter()
            avg += ((end - start)*1000)
            # print(str((end - start)*1000) + " milliseconds")
            # board.vis()
            board.reset()
print("total: " + str(time.perf_counter() - total))
avg /= games_played
print("per game: " + str(avg) + " milliseconds")
print(games_played)
