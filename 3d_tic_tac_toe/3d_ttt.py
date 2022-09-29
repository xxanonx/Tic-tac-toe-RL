# created by McKinley James Pollock on 9/10/22
# made for research purposes and fun

import numpy as np
from numba import jit
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Flatten
import random, time
import matplotlib.pyplot as plt
from dataclasses import dataclass

"""config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)"""

IN_A_ROW = 3
EMPTY_BOARD = np.zeros((IN_A_ROW, IN_A_ROW, IN_A_ROW), dtype='int8')
EMPTY_BOARD_MOVES = np.zeros((IN_A_ROW, IN_A_ROW), dtype='float16')
ONES = np.ones((IN_A_ROW, IN_A_ROW, IN_A_ROW), dtype='int8')
NEG_ONES = (np.ones((IN_A_ROW, IN_A_ROW, IN_A_ROW), dtype='int8') * -1)

random.seed(time.time_ns())
actual_play = False

# dna related stuff
activate = ['sigmoid', 'relu', 'selu', 'linear', 'gelu']
losses = ['categorical_crossentropy', 'binary_crossentropy', 'mean_absolute_error',
          'mean_squared_error', 'sparse_categorical_crossentropy']
# layers = ['dense']
minimum = (pow(IN_A_ROW, 3) * 2)


@dataclass
class DnaStrand:
    layer: str
    size: int
    activation: str


class BoardEnv:

    def __init__(self):
        self.board = np.copy(EMPTY_BOARD)
        self.board_history_O = []
        self.board_history_X = []
        self.move_history_O = []
        self.move_history_X = []
        self.current_board = np.copy(self.board)
        self.game_over = False
        self.whose_turn = True
        self.winner = 0
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
        self.board_history_O = []
        self.board_history_X = []
        self.move_history_O = []
        self.move_history_X = []
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

    def make_move(self, x: int, y: int, actor_answer=None):
        move_made: bool = False
        actor_decides = actor_answer is not None
        self.current_board = np.copy(self.board)

        if actor_decides:
            armax = tf.argmax(actor_answer).numpy()
            x = armax.argmax()
            y = armax.max()

        layer_num = 0
        for layer in self.board:
            if move_made:
                break
            if layer[y][x] == 0:
                # move2 is used for move data
                move2 = np.copy(EMPTY_BOARD_MOVES)
                move2[y][x] = 1
                if self.whose_turn:
                    self.board[layer_num][y][x] = 1
                    self.board_history_X.append(np.copy(self.board))
                    self.move_history_X.append(np.copy(move2))
                    move_made = True
                else:
                    self.board[layer_num][y][x] = -1
                    self.board_history_O.append(np.copy(self.board))
                    self.move_history_O.append(np.copy(move2))
                    move_made = True
            layer_num += 1
        # return false if the move can't be made
        return move_made

    def get_state(self, human=False, random_play=False, verbose=False, actor_answer=None):
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

            if self.make_move(final_move[0], final_move[1], actor_answer):
                if verbose:
                    print(f"{final_move} is acceptable")
                break
            else:
                if not random_play and actor_answer is not None:
                    armax = tf.argmax(actor_answer).numpy()
                    actor_answer[armax.max()][armax.argmax()] = -1
                if verbose:
                    print(f"it seems that {final_move} is not legal")

        done, self.winner =self.look_for_win()
        if done:
            if verbose:
                print(self.board)
                if self.winner == -1:
                    print("O's WON!!!")
                elif self.winner == 1:
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
        self.Actor_model = Sequential()
        self.do_not_init_a = False
        self.dna = []
        self.loss = ""
        if not self.do_not_init_a:
            self.init_actor()
            self.do_not_init_a = True

    def init_actor(self):
        # Actor trained on wins
        self.dna = []

        # creation of randomly generated actor
        number_of_layers = random.randint(4, 14)
        for layer in range(number_of_layers):
            if layer == 0:
                # first layer is always the same
                size = minimum
            elif layer == number_of_layers - 1:
                # last layer is always the same
                size = 9
            else:
                # any other layer
                size = random.randint(((minimum / 2) - (layer * 2)), minimum)
            self.dna.append(DnaStrand('Dense', size, random.choice(activate)))
        self.loss = random.choice(losses)

        self.make_actor()

    def mutate_actor(self, override_max_number_of_mutations=4):
        del self.Actor_model
        self.Actor_model = Sequential()
        max_size = len(self.dna)
        what_layers_mutated = []
        # how many mutations
        for mutation in range(random.randint(1, override_max_number_of_mutations)):
            where = random.randint(1, (max_size - 2))
            type_of_mutation = random.randint(0, 3)
            if type_of_mutation == 0:
                # change layer size
                self.dna[where].size = random.randint(((minimum / 2) - (where * 2)), minimum)
            elif type_of_mutation == 1:
                # change activation
                self.dna[where].activation = random.choice(activate)
            elif type_of_mutation == 2:
                # delete strand
                self.dna.pop(where)
            elif type_of_mutation == 3:
                # add strand
                size = random.randint(((minimum / 2) - (where * 2)), minimum)
                self.dna.insert(where, DnaStrand('Dense', size, random.choice(activate)))
            elif type_of_mutation == 4:
                # change loss
                self.loss = random.choice(losses)
            # new size in case dna got bigger or smaller
            max_size = len(self.dna)
        self.make_actor()

    def make_actor(self):
        number_of_layers = len(self.dna)
        self.Actor_model.add(Dense(self.dna[0].size, input_shape=(IN_A_ROW, IN_A_ROW, IN_A_ROW),
                                   activation=self.dna[0].activation))
        strand_num = 0
        for strand in self.dna:
            if strand_num != 0:
                if strand.layer.lower() == 'dense':
                    self.Actor_model.add(Dense(strand.size, activation=strand.activation))
            if strand_num == int(number_of_layers / 2):
                # halfway through flatten
                self.Actor_model.add(Flatten())
            strand_num += 1

        self.Actor_model.add(Reshape((3, 3)))
        self.Actor_model.compile(
            optimizer='adam',
            loss=self.loss,
            metrics=['accuracy']
        )
        self.Actor_model.summary()
        # self.Actor_model.fit(x=np.array([np.copy(EMPTY_BOARD)]))

    def teach_actor(self, states, moves):
        div = int((len(states) * 0.7))
        train_x = np.copy(states[:div]).astype('float16')
        validation_x = np.copy(states[div:]).astype('float16')
        train_y = np.copy(moves[:div]).astype('float16')
        validation_y = np.copy(moves[div:]).astype('float16')

        print(train_x.dtype)
        print(train_y.dtype)
        print(train_x.shape)
        print(train_y.shape)

        print(train_x)
        print(train_y)

        self.Actor_model.fit(train_x, train_y, validation_data=(validation_x, validation_y), epochs=20)
        self.Actor_model.save(
            f'/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/3D_tic_tac_toe/save_models/TTT3D_actor_{IN_A_ROW}iar')

    def load_model(self):
        self.Actor_model = load_model(
            f'/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/3D_tic_tac_toe/save_models/TTT3D_actor_{IN_A_ROW}iar')
        self.do_not_init_a = True


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
                '''print(row)
                print('player won')'''
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


def reward_previous_moves(moves, multiplier):
    moves.reverse()
    for move in moves:
        move *= multiplier
        multiplier /= 2
    moves.reverse()
    return moves.copy()


list_of_games = []
list_of_moves = []
list_of_boards = []
randomness = 1.0
for i in range(500):
    list_of_boards.append(BoardEnv())

actor = Player()

actual_play = True
first_run = True
set_amount_of_games = 10000
total = time.perf_counter()
for gen in range(20):
    games_played = 0
    prog_bar = tqdm(total=set_amount_of_games)
    while games_played < set_amount_of_games:
        for board in list_of_boards:
            chance_of_random = random.randint(0, int(randomness * 100))
            if chance_of_random != 0 or first_run:
                board.get_state(False, True)
            else:
                move = actor.Actor_model(np.copy(board.board))
                board.get_state(False, False, actor_answer=move)
            if board.game_over:
                temp_moves_X = []
                temp_moves_O = []
                if board.winner == -1:
                    temp_moves_O = reward_previous_moves(board.move_history_O, 1)
                    temp_moves_X = reward_previous_moves(board.move_history_X, -1)
                elif board.winner == 1:
                    temp_moves_O = reward_previous_moves(board.move_history_O, -1)
                    temp_moves_X = reward_previous_moves(board.move_history_X, 1)
                else:
                    temp_moves_O = reward_previous_moves(board.move_history_O, -1)
                    temp_moves_X = reward_previous_moves(board.move_history_X, -1)
                for i in board.board_history_O:
                    i *= -1
                list_of_moves.extend(temp_moves_X)
                list_of_moves.extend(temp_moves_O)
                list_of_games.extend(board.board_history_X.copy())
                list_of_games.extend(board.board_history_O.copy())
                board.reset()
                prog_bar.update(1)
                if games_played >= set_amount_of_games:
                    break
    if first_run:
        first_run = False
    actor.teach_actor(np.array(list_of_games), np.array(list_of_moves))
    randomness *= 0.95
time.sleep(0.1)
end = (time.perf_counter() - total)
print("total: " + str(end))
end /= (games_played * 20)
print("per game: " + str(end) + " milliseconds")
print(games_played * 20)
