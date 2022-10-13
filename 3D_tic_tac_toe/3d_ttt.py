# created by McKinley James Pollock on 9/10/22
# made for research purposes and fun

import numpy as np
from numba import jit
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout
import random, time, pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

IN_A_ROW = 3
MAX_LEN = 500000
EMPTY_BOARD = np.zeros((IN_A_ROW, IN_A_ROW, IN_A_ROW), dtype='int8')
EMPTY_BOARD_MOVES = np.zeros((IN_A_ROW, IN_A_ROW), dtype='float16')
ONES = np.ones((IN_A_ROW, IN_A_ROW, IN_A_ROW), dtype='int8')
NEG_ONES = (np.ones((IN_A_ROW, IN_A_ROW, IN_A_ROW), dtype='int8') * -1)

random.seed(time.time_ns())
actual_play = False
count_of_bypasses = 0

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

    def __init__(self, seed=0):
        self.board = np.copy(EMPTY_BOARD)
        self.board_history_O = []
        self.board_history_X = []
        self.move_history_O = []
        self.move_history_X = []
        self.previous_board = np.copy(self.board)
        self.game_over = False
        self.whose_turn = True
        self.winner = 0
        self.random_ = random.Random()
        self.random_.seed(seed)
        self.how_many_times_won = 0
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
        self.previous_board = np.copy(self.board)

        self.game_over = False
        self.how_many_times_won = 0
        if actual_play:
            games_played += 1
        # self.whose_turn = not self.whose_turn

    def vis(self):
        visualize3d(self.board)

    def look_for_win(self):
        # if no zeros are found the game is over, this can help tell if it's a tie
        board_filled = self.board.all()
        winner = 0
        temp_times_won = 0
        temp_game_over_buffer = []
        temp_winner_buffer = []
        # look at board from bottom to top, side to side, and other side to side
        if not self.game_over:
            # checking layers
            for orientation in [self.board,
                                np.moveaxis(self.board, 0, -1),
                                np.moveaxis(self.board, -1, 0)]:
                for board_layer in orientation:
                    if board_layer.any():
                        temp_game_over, temp_winner, temp_times_won = check_layer_for_win(board_layer)
                        if temp_times_won > self.how_many_times_won:
                            self.how_many_times_won += temp_times_won
                        temp_game_over_buffer.append(temp_game_over)
                        temp_winner_buffer.append(temp_winner)
            if self.how_many_times_won >= 3:
                self.how_many_times_won /= 3
            # special diagonals
            # self.board.diagonal() comes out as 3x3 so already single layer
            for diag in [self.board.diagonal(), np.fliplr(self.board).diagonal()]:
                if diag.any():
                    temp_game_over, temp_winner, temp_times_won = check_layer_for_win(diag, True)
                    self.how_many_times_won += temp_times_won
                    temp_game_over_buffer.append(temp_game_over)
                    temp_winner_buffer.append(temp_winner)

            for game_over in temp_game_over_buffer:
                self.game_over = game_over
                if self.game_over:
                    for winner_ in temp_winner_buffer:
                        if winner_ != 0:
                            winner = winner_
                            break
                    break

        if board_filled and not self.game_over:
            self.game_over = True
            winner = 0
        return self.game_over, winner

    def make_move(self, x: int, y: int, actor_answer=None, verbose=False):
        move_made: bool = False
        actor_decides = actor_answer is not None
        self.previous_board = np.copy(self.board)

        if actor_decides:
            armax = tf.argmax(actor_answer).numpy()
            x = armax.argmax()
            y = armax.max()

        layer_num = 0
        for layer in self.board:
            if move_made or self.game_over:
                break
            if layer[y][x] == 0:
                # move2 is used for move data
                move2 = np.copy(EMPTY_BOARD_MOVES)
                move2[y][x] = 1
                if self.whose_turn:
                    self.board[layer_num][y][x] = 1
                    self.board_history_X.append(np.copy(self.previous_board))
                    self.move_history_X.append(np.copy(move2))
                    move_made = True
                else:
                    self.board[layer_num][y][x] = -1
                    self.board_history_O.append(np.copy(self.previous_board))
                    self.move_history_O.append(np.copy(move2))
                    move_made = True
            layer_num += 1
        if move_made and verbose:
            print(f"[{x}, {y}] is acceptable")
        # return false if the move can't be made
        return move_made

    def get_state(self, human=False, random_play=False, verbose=False, actor_answer=None):
        # Whose turn matters and human matters
        global count_of_bypasses
        input_max = (IN_A_ROW - 1)
        error_in_row = 0
        actor_bypass = False
        temp_list_of_failed_moves = []
        if self.game_over:
            return
        if verbose and human:
            visualize3d(self.board)
        while True:
            if self.game_over:
                break
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
                if random_play or actor_bypass:
                    final_move[0] = self.random_.randint(0, input_max)
                    final_move[1] = self.random_.randint(0, input_max)

            if self.make_move(final_move[0], final_move[1], actor_answer, verbose):
                if verbose:
                    if actor_answer is not None:
                        print(actor_answer)
                    else:
                        print(f"{final_move} is acceptable")
                break
            else:
                error_in_row += 1
                temp_move = np.copy(EMPTY_BOARD_MOVES)
                if not random_play and actor_answer is not None:
                    armax = tf.argmax(actor_answer).numpy()
                    temp_move[armax.max()][armax.argmax()] = -1
                    actor_answer[armax.max()][armax.argmax()] = -1
                else:
                    temp_move[final_move[1]][final_move[0]] = -1
                temp_list_of_failed_moves.append(np.copy(temp_move))

                if error_in_row > 5:
                    if actor_answer is not None:
                        actor_answer = None
                        actor_bypass = True
                    count_of_bypasses += 1
                if verbose:
                    print(f"it seems that {final_move} is not legal")

        done, self.winner = self.look_for_win()
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
        return temp_list_of_failed_moves


class Player:
    def __init__(self, sign=1, dont_init=False):
        self.sign = sign
        self.score = 0
        self.Actor_model = Sequential()
        self.do_not_init_a = dont_init
        self.dna = []
        self.loss = ""
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []
        self.teaching_with_pickle = False
        if not self.do_not_init_a:
            self.init_actor()
            self.do_not_init_a = True
        else:
            self.load_model()

    def init_actor(self):
        # Actor trained on wins
        if not self.do_not_init_a:
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
                self.Actor_model.add(Dropout(0.25))
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
        train_x = np.copy(states[:div]).astype('float32')
        validation_x = np.copy(states[div:]).astype('float32')
        train_y = np.copy(moves[:div]).astype('float32')
        validation_y = np.copy(moves[div:]).astype('float32')

        print(train_x.dtype)
        print(train_y.dtype)
        print(train_x.shape)
        print(train_y.shape)
        print(validation_x.shape)
        print(validation_y.shape)

        '''print(train_x)
        print(train_y)'''

        temp_fitting_history = self.Actor_model.fit(train_x, train_y, validation_data=(validation_x, validation_y),
                                                    epochs=4, shuffle=True)
        self.loss_history.extend(temp_fitting_history.history['loss'])
        self.accuracy_history.extend(temp_fitting_history.history['accuracy'])
        self.val_loss_history.extend(temp_fitting_history.history['val_loss'])
        self.val_accuracy_history.extend(temp_fitting_history.history['val_accuracy'])
        self.Actor_model.save(
            f'/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/3D_tic_tac_toe/save_models/TTT3D_actor_{IN_A_ROW}iar')
        self.write_pickles(states, moves)

    def load_model(self):
        print("loading model!")
        self.Actor_model = load_model(
            f'/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/3D_tic_tac_toe/save_models/TTT3D_actor_{IN_A_ROW}iar')
        self.do_not_init_a = True

    def teach_on_pickle(self):
        self.teaching_with_pickle = True
        with open('3d_ttt_learning_states.pickle', 'rb') as re:
            pickle_states = pickle.load(re)
        with open('3d_ttt_learning_moves.pickle', 'rb') as re:
            pickle_moves = pickle.load(re)
        self.teach_actor(pickle_states, pickle_moves)
        self.teaching_with_pickle = False

    def write_pickles(self, pickle_states, pickle_moves):
        if not self.teaching_with_pickle:
            with open('3d_ttt_learning_states.pickle', 'wb') as wr:
                pickle.dump(pickle_states, wr)
            with open('3d_ttt_learning_moves.pickle', 'wb') as wr:
                pickle.dump(pickle_moves, wr)

    def vis_training(self):
        plt.plot(self.loss_history, label='loss_history')
        plt.plot(self.accuracy_history, label='accuracy_history')
        plt.plot(self.val_loss_history, label='val_loss_history')
        plt.plot(self.val_accuracy_history, label='val_accuracy_history')
        plt.legend(loc='lower right')
        plt.show()


# for 2D
def check_layer_for_win(layer2d: np.ndarray, just_diag= False):
    game_over = False
    winner = 0
    times_won = 0
    # Checking horizontal and vertical
    if not just_diag:
        for board2d in [layer2d, np.copy(layer2d.transpose())]:
            for row in board2d:
                '''if game_over:
                    break'''
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
                        times_won += 1
                        break

    '''[[0. 0. 0.]
        [0. 0. 0.]      Board/layer
        [0. 0. 0.]]'''
    # Checking Diagonal
    for diag in [layer2d.diagonal(), np.flipud(layer2d).diagonal()]:
        '''if game_over:
            break'''
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
                times_won += 1

    return game_over, winner, times_won


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
        multiplier *= 0.8
    moves.reverse()
    return moves.copy()


list_of_games = deque(maxlen=MAX_LEN)
list_of_moves = deque(maxlen=MAX_LEN)
list_of_master_moves = deque(maxlen=50000)
list_of_master_games = deque(maxlen=50000)
list_of_failed_moves = deque(maxlen=50000)
list_of_failed_games = deque(maxlen=50000)
list_of_boards = []
randomness = 100
tempSeed = time.time_ns()
for i in range(500):
    list_of_boards.append(BoardEnv(tempSeed + i))

actor = Player(dont_init=True)

actual_play = True
first_run = True
I_want_to_play = True
I_want_to_watch = False
set_amount_of_games = 20000
total = time.perf_counter()
for gen in range(20):
    games_played = 0
    print(gen)
    prog_bar = tqdm(total=set_amount_of_games)
    if not I_want_to_play:
        while games_played < set_amount_of_games:
            for board in list_of_boards:
                chance_of_random = random.randint(0, int(randomness * 100))
                if chance_of_random != 0 or first_run:
                    # any bad moves go to list_of_failed_moves
                    temp_list_of_failed_moves = board.get_state(False, True)
                else:
                    if board.whose_turn:
                        move = actor.Actor_model(np.array([np.copy(board.board)], dtype='float32'))[0].numpy()
                    else:
                        move = actor.Actor_model((np.array([np.copy(board.board)], dtype='float32')) * -1)[0].numpy()
                    temp_list_of_failed_moves = board.get_state(False, False, actor_answer=move)
                    # If I want to watch and its board 0 of the 500
                    if I_want_to_watch and board.random_ == list_of_boards[0].random_ and board.game_over:
                        print(move)
                        board.vis()

                if len(temp_list_of_failed_moves) > 0:
                    for fail in temp_list_of_failed_moves:
                        list_of_failed_games.append(np.copy(board.previous_board))
                        list_of_failed_moves.append(np.copy(fail))

                if board.game_over:
                    temp_moves_X = []
                    temp_moves_O = []

                    if board.winner == -1:
                        temp_moves_O = reward_previous_moves(board.move_history_O, board.how_many_times_won)
                        temp_moves_X = reward_previous_moves(board.move_history_X, -1)
                    elif board.winner == 1:
                        temp_moves_O = reward_previous_moves(board.move_history_O, -1)
                        temp_moves_X = reward_previous_moves(board.move_history_X, board.how_many_times_won)
                    else:
                        if random.randint(0, 10) <= 6:
                            board.reset()
                            prog_bar.update(1)
                            continue
                        temp_moves_O = reward_previous_moves(board.move_history_O, -1)
                        temp_moves_X = reward_previous_moves(board.move_history_X, -1)
                    tempBoard_hist_O = []
                    for i in board.board_history_O.copy():
                        tempBoard_hist_O.append(i * -1)
                    if random.randint(0, 50) != 0:
                        if not board.board_history_X[0][0].any():
                            temp_moves_X.pop(0)
                            board.board_history_X.pop(0)
                        elif not tempBoard_hist_O[0].any():
                            temp_moves_O.pop(0)
                            tempBoard_hist_O.pop(0)

                    # Hopefully actor can learn from losing maybe to go where it lost to
                    if board.winner == -1:
                        temp_moves_X.append(np.copy(temp_moves_O[-1]))
                        board.board_history_X.append(np.copy(board.board_history_X[-1]))
                    elif board.winner == 1:
                        temp_moves_O.append(np.copy(temp_moves_X[-1]))
                        tempBoard_hist_O.append(np.copy(tempBoard_hist_O[-1]))

                    # Because up to this point we have a python list of np.arrays,
                    # we shall iterate over the list and copy items one at a time
                    # perhaps this will help with the weird moves the actor currently makes
                    '''list_of_moves.extend(temp_moves_X.copy())
                    list_of_moves.extend(temp_moves_O.copy())
                    list_of_games.extend(board.board_history_X.copy())
                    list_of_games.extend(tempBoard_hist_O.copy())'''

                    if False and board.how_many_times_won > 1:
                        # used for troubleshooting
                        print(board.board_history_X)
                        print(temp_moves_X)
                        print(tempBoard_hist_O)
                        print(temp_moves_O)
                        if board.winner == 1:
                            print("RED")
                        elif board.winner == -1:
                            print("BLUE")
                        print(board.how_many_times_won)
                        board.vis()
                        '''for i in (board.board_history_O, board.board_history_X):
                            for ii in i:
                                visualize3d(ii)'''

                    if (((len(board.board_history_X) - len(temp_moves_X)) != 0) or
                            ((len(tempBoard_hist_O) - len(temp_moves_O)) != 0)):
                        print("board and move data not the same length!")
                    for moves in (temp_moves_X, temp_moves_O):
                        for single_move in moves:
                            list_of_moves.append(np.copy(single_move))

                    for games in (board.board_history_X, tempBoard_hist_O):
                        for single_round in games:
                            list_of_games.append(np.copy(single_round))

                    if (len(list_of_games) - len(list_of_moves)) != 0:
                        print("Training data not the same length!")

                    if board.how_many_times_won > 1:
                        if board.winner == 1:
                            for move in temp_moves_X:
                                list_of_master_moves.append(np.copy(move))
                            for round_ in board.board_history_X:
                                list_of_master_games.append(np.copy(round_))

                        elif board.winner == -1:
                            for move in temp_moves_O:
                                list_of_master_moves.append(np.copy(move))
                            for round_ in tempBoard_hist_O:
                                list_of_master_games.append(np.copy(round_))

                    board.reset()
                    prog_bar.update(1)
                    if games_played >= set_amount_of_games:
                        break

        if first_run:
            first_run = False
        print("difference in total length:" + str(len(list_of_games) - len(list_of_moves)))

        actor.teach_actor(np.concatenate((np.array(list_of_failed_games),
                                          np.array(list_of_master_games),
                                          np.array(list_of_games))),
                          np.concatenate((np.array(list_of_failed_moves),
                                          np.array(list_of_master_moves),
                                          np.array(list_of_moves))))
    else:
        against_me = list_of_boards[0]
        against_me.reset()
        while not against_me.game_over:
            if against_me.whose_turn:
                move = actor.Actor_model(np.array([np.copy(against_me.board)], dtype='float32'))[0].numpy()
                print(move)
                against_me.get_state(False, False, actor_answer=move, verbose=True)
            else:
                if I_want_to_watch:
                    against_me.get_state(random_play=True, verbose=True)
                else:
                    against_me.get_state(True, verbose=True)
    randomness *= 0.98
time.sleep(0.1)
end = (time.perf_counter() - total)
print("total: " + str(end))
end /= (games_played * 20)
print("per game: " + str(end) + " milliseconds")
print(games_played * 20)
print("count_of_bypasses: " + str(count_of_bypasses))
actor.vis_training()
