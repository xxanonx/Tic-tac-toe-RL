#   started 03/31/2021
#   goal create tic tac toe game and have agents play it

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Flatten
import random, time
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


class player:
    def __init__(self, sign=1):
        self.sign = sign
        self.score = 0
        self.opponent_score = 0
        self.Critic_model = Sequential()
        self.Actor_model = Sequential()
        self.do_not_init = False

    def init_critic(self):
        # Critic Model
        # Looks at state and thinks its a good state for actor or not
        self.Critic_model.add(Dense(9, input_shape=(2, 3, 3), activation='relu'))
        self.Critic_model.add(Flatten())
        self.Critic_model.add(Dense(9, activation='relu'))
        self.Critic_model.add(Dense(9, activation='relu'))
        self.Critic_model.add(Dense(1, activation='relu'))
        self.Critic_model.compile(
            optimizer='adam',
            loss='huber_loss',
            metrics=['accuracy']
        )
        self.Critic_model.summary()

    def init_actor(self):
        # Actor trained on wins
        self.Actor_model.add(Dense(9, input_shape=(2, 3, 3), activation='relu'))
        self.Actor_model.add(Dense(9, activation='relu'))
        self.Actor_model.add(Flatten())
        self.Actor_model.add(Dense(9, activation='relu'))
        self.Actor_model.add(Dense(9, activation='hard_sigmoid'))
        self.Actor_model.add(Reshape((3, 3)))
        self.Actor_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.Actor_model.summary()

    def teach_critc(self, state, value):
        if not self.do_not_init: self.init_critic()
        div = int((len(state) * 0.7))
        train_x = np.copy(state[:div]).astype('int')
        validation_x = np.copy(state[div:]).astype('int')
        train_y = np.copy(value[:div]).astype('float64')
        validation_y = np.copy(value[div:]).astype('float64')

        print(train_x.dtype)
        print(train_y.dtype)
        print(train_x.shape)
        print(train_y.shape)

        print(train_x)
        print(train_y)

        self.Critic_model.fit(train_x, train_y,  validation_data=(validation_x, validation_y), epochs=10)
        self.Critic_model.save('/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/save_models/TTT2D_Critic')

    def teach_actor(self, state, move):
        if not self.do_not_init: self.init_actor()
        div = int((len(state) * 0.7))
        train_x = np.copy(state[:div]).astype('int')
        validation_x = np.copy(state[div:]).astype('int')
        train_y = np.copy(move[:div]).astype('int')
        validation_y = np.copy(move[div:]).astype('int')

        print(train_x.dtype)
        print(train_y.dtype)
        print(train_x.shape)
        print(train_y.shape)

        print(train_x)
        print(train_y)

        self.Actor_model.fit(train_x, train_y, validation_data=(validation_x, validation_y), epochs=15)
        self.Actor_model.save('/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/save_models/TTT2D_actor')

    def load_models(self):
        self.Critic_model = load_model(
            '/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/save_models/TTT2D_Critic')
        self.Actor_model = load_model(
            '/mnt/96a66be0-609e-43bd-a076-253e3c725b17/Python/RL testing/save_models/TTT2D_actor')
        self.do_not_init = True


class board_env:
    def __init__(self):
        self.p1 = player(-1)
        # For actor
        self.round_buffer = []
        self.move_buffer = []
        self.actor_training = []
        self.actor_moves = []
        # For critic
        self.recorded_games = []
        self.recorded_scores = []
        # other
        self.games_played = -1
        self.board = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        self.previous_board = np.copy(self.board)
        self.current_board = np.copy(self.board)
        self.game_over = False
        self.whose_turn = True
        # -1 = O and 1 = X
        # false = 0 and true = X
        self.empty_board = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
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
        self.empty_board = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        self.previous_board = np.copy(self.board)
        self.game_over = False
        self.games_played += 1
        # self.whose_turn = not self.whose_turn

    def look_for_win(self):
        # if no zeros are found the game is over, this can tell if its a tie
        board_filled = self.board.all()
        in_a_row = 0
        # Checking horizontal
        # print("Checking horizontal")
        for row in self.board:
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
                    self.game_over = True
                    return True, row[0]
                    break

        # Checking vertical
        # print("Checking vertical")
        for col in self.board.transpose():
            in_a_row = 0
            if col.all():
                # if line doesn't have zeros
                for valc in col:
                    # double check that all values are the same
                    if int(valc) != int(col[0]):
                        break
                    else:
                        in_a_row += 1

                if in_a_row == 3:
                    self.game_over = True
                    return True, col[0]
                    break

        # Checking Diagonal
        # diag_win = False
        # print("Checking Diagonal")
        ULC_2_LRC = self.board.diagonal()
        if ULC_2_LRC.all():
            in_a_row = 0
            # if line doesn't have zeros
            for vald1 in ULC_2_LRC:
                # double check that all values are the same
                if int(vald1) != int(ULC_2_LRC[0]):
                    break
                else:
                    in_a_row += 1

            # diag_win = True
            if in_a_row == 3:
                self.game_over = True
                return True, ULC_2_LRC[0]

        # print("Checking Diagonal 2")
        LLC_2_URC = np.flipud(self.board).diagonal()
        if LLC_2_URC.all():
            in_a_row = 0
            # if line doesn't have zeros
            for vald2 in LLC_2_URC:
                # double check that all values are the same
                if int(vald2) != int(LLC_2_URC[0]):
                    break
                else:
                    in_a_row += 1
            # diag_win = True
            if in_a_row == 3:
                self.game_over = True
                return True, LLC_2_URC[0]

        if board_filled and not self.game_over:
            self.game_over = True
            return True, 0
        else:
            return False, 0

    def value_of_board(self, is_win, is_turn, symbol=1):
        # print("Checking horizontal")
        rough_value = 0
        win_found = False
        if self.board.any():
            for row in self.board:
                in_a_row = 0
                # print(row)
                if (row.any() and not is_win and not row.all()) or (is_win and row.all()):
                    for valr in row:
                        # double check that all values are the same
                        if int(valr) != symbol:
                            if int(valr) != 0:
                                in_a_row -= 1
                                # break
                        else:
                            in_a_row += 1

                    if (not is_turn and (-1 > in_a_row or in_a_row > 1)) or (is_turn and (in_a_row >= 2)):
                        rough_value += in_a_row
                        if in_a_row == 3:
                            win_found = True


            # Checking vertical
            # print("Checking vertical")
            for col in self.board.transpose():
                in_a_row = 0
                # print(row)
                if (col.any() and not is_win and not col.all()) or (is_win and col.all()):
                    for valc in col:
                        # double check that all values are the same
                        if int(valc) != symbol:
                            if int(valc) != 0:
                                in_a_row -= 1
                                # break
                        else:
                            in_a_row += 1

                    if (not is_turn and (-1 > in_a_row or in_a_row > 1)) or (is_turn and (in_a_row >= 2)):
                        rough_value += in_a_row
                        if in_a_row == 3:
                            win_found = True

            # Checking Diagonal
            # diag_win = False
            # print("Checking Diagonal")
            ULC_2_LRC = self.board.diagonal()
            if (ULC_2_LRC.any() and not is_win and not ULC_2_LRC.all()) or (is_win and ULC_2_LRC.all()):
                in_a_row = 0
                for vald1 in ULC_2_LRC:
                    if int(vald1) != symbol:
                        if int(vald1) != 0:
                            in_a_row -= 1
                            # break
                    else:
                        in_a_row += 1

                if (not is_turn and (-1 > in_a_row or in_a_row > 1)) or (is_turn and (in_a_row >= 2)):
                    rough_value += in_a_row
                    if in_a_row == 3:
                        win_found = True

            # print("Checking Diagonal 2")
            LLC_2_URC = np.flipud(self.board).diagonal()
            if (LLC_2_URC.any() and not is_win and not LLC_2_URC.all()) or (is_win and LLC_2_URC.all()):
                in_a_row = 0
                for vald2 in LLC_2_URC:
                    if int(vald2) != symbol:
                        if int(vald2) != 0:
                            in_a_row -= 1
                            # break
                    else:
                        in_a_row += 1

                if (not is_turn and (-1 > in_a_row or in_a_row > 1)) or (is_turn and (in_a_row >= 2)):
                    rough_value += in_a_row
                    if in_a_row == 3:
                        win_found = True

            # final estimate
            rough_value /= np.count_nonzero(self.board)
            if win_found:
                rough_value += 0.4

            if rough_value > 1.0:
                rough_value = 1.0

        return rough_value

    def make_move(self, x, y):
        if self.board[y][x] == 0:
            # for actor
            if not self.whose_turn:
                # for critic
                self.current_board = np.copy(self.board)
                self.recorded_games.append([self.current_board, self.previous_board])
                self.recorded_scores.append(self.value_of_board(self.game_over, self.whose_turn, symbol=-1))

                # for actor
                move2 = np.copy(self.empty_board)
                move2[y][x] = 1
                self.move_buffer.append(move2)
                self.round_buffer.append([self.current_board, self.previous_board])
                # print([self.board, self.previous_board])
                # print(move2)

            if self.whose_turn:
                self.board[y][x] = 1
            else:
                self.board[y][x] = -1
                self.previous_board = np.copy(self.board)
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
                if ("," in move) and (5 >= len(move) >= 3) \
                        and ((move1[0].isdigit() or move1[0] == "-1") and (move1[1].isdigit() or move1[1] == "-1")):
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
            while True:
                self.current_board = np.copy(self.board)
                if verbose:
                    # print([self.current_board, self.previous_board])
                    # print(self.p1.Critic_model.predict(np.array([[self.current_board, self.previous_board]])))
                    pass
                move = [0,0]
                ai_can_skip = False
                if random_play:
                    random.seed(time.time_ns())
                    move[0] = random.randint(-1, 1)
                    move[1] = random.randint(-1, 1)
                else:
                    pred = (self.p1.Actor_model.predict(np.array([[self.current_board, self.previous_board]]))[0])
                    if verbose:
                        print('Algorithm playing!')
                        print(pred)
                    if pred.any():
                        ai_y = 0
                        max = pred.ravel().max()
                        for row in pred:
                            row_max = row[int(tf.argmax(row))]
                            if row_max == max:
                                ai_can_skip = True
                                move = [(int(tf.argmax(row))), ai_y]
                                break
                            ai_y += 1

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
                        if not random_play and ai_can_skip:
                            print(f"{move} is acceptable")
                        break

        done, winner_ = self.look_for_win()
        if done:
            if self.p1.sign == winner_:
                self.p1.score += 1
            else:
                self.p1.opponent_score += 1
            if winner_ == -1:
                self.actor_training.extend(self.round_buffer)
                self.actor_moves.extend(self.move_buffer)
            self.round_buffer.clear()
            self.move_buffer.clear()
            if not verbose:
                self.reset()

        # end of turn
        self.whose_turn = not self.whose_turn


# BUFFER ZONE

b1 = board_env()
# print(b1.board)
learning = False
how_much_off = []
while True:
    if b1.games_played > 100000 or True:
        if learning:
            print(f"{b1.p1.score} V.S. {b1.p1.opponent_score}")
            b1.p1.teach_critc(b1.recorded_games, b1.recorded_scores)
            b1.p1.teach_actor(b1.actor_training, b1.actor_moves)
            learning = False

        else:
            b1.p1.load_models()

        if b1.games_played > 1:
            b1.get_state(human=b1.whose_turn, random_play=False, verbose=True)
        else:
            b1.get_state(human=b1.whose_turn, random_play=True, verbose=True)

        game_over, winner = b1.look_for_win()
        # AI_predicted = b1.
        '''Calculated_o = b1.value_of_board(game_over, b1.whose_turn, -1)
        how_much_off.append(Calculated_o - AI_predicted[0][0])
        print(f"Calculated X Value: {b1.value_of_board(game_over, not b1.whose_turn)}, Calculated O Value: {Calculated_o}")
        print(f"Ai Value for O's: {AI_predicted[0][0]}, it is {how_much_off[-1]} off")'''

        if game_over:
            print(b1.board)
            if winner == -1:
                print("Computer WON!!!")
            elif winner == 1:
                print("Person WON!!!!")
            else:
                print("DRAW")
            b1.reset()

            '''plt.plot(how_much_off)
            plt.show()
            how_much_off = []'''

    else:
        b1.get_state(human=False, random_play=True)
        learning = True
        # print(b1.board)
        # time.sleep(1)
