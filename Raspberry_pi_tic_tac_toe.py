#   started 08/18/2021
#   goal: use tic tac toe game that is played on computer to work on raspberry pi using TF Lite

import numpy as np
import board
import neopixel
import tflite_runtime.interpreter as tflite
# from tensorflow import lite as tflite
import random, time
from gpiozero import LED, Button
import matplotlib.pyplot as plt
import socket

go_led = LED(23)
# pixel = neopixel.NeoPixel(board.D18, 9, pixel_order=neopixel.GRB)
_b_xno = Button(2)
_b_xz = Button(3)
_b_xo = Button(4)
_b_yno = Button(17)
_b_yz = Button(27)
_b_yo = Button(22)

x_buttons = [_b_xno, _b_xz,_b_xo]
y_buttons = [_b_yno, _b_yz, _b_yo]

TERMINAL_INPUT = False
'''X_COLOR = (255,0,0)
O_COLOR = (0,0,255)'''

host = socket.gethostname()  # get local machine name
port = 8080  # Make sure it's within the > 1024 $$ <65535 range

s = socket.socket()
s.connect((host, port))


def map_range(x, from_low=-1, from_high=1, to_low=0, to_high=255):
    y = to_low + ((to_high - to_low) / (from_high - from_low)) * (x - from_low) 
    return y


def visualize_board(boardy, game_done, human=True):
    vis_board = np.copy(boardy).ravel()
    # repeat = 1
    s.send(vis_board)
    data = s.recv(1024).decode('utf-8')
    print('Received from server: ' + data)
    s.close()
    '''if game_done:
        repeat = 6
    for r in range(repeat):
        if game_done and repeat % 2 == 0:
            pixel.fill((0,0,0))
            time.sleep(1)
        
        else:
            for box in range(vis_board.size):
                pix = box
                if box == 3:
                    pix = 5
                elif box == 5:
                    pix = 3
                
                elif vis_board[box] == -1:
                    pixel[pix] = X_COLOR
                elif vis_board[box] == 1:
                    pixel[pix] = O_COLOR
                else:
                    pixel[pix] = (0,0,0)'''
                    
    # pixel.show()
    if not game_done:
        x_in = False
        y_in = False
        while True:
            if not x_in:
                xnum = -1
                for xbut in x_buttons:
                    if xbut.is_pressed:
                        x_in = True
                        break
                    xnum += 1
            if not y_in:
                ynum = -1
                for ybut in y_buttons:
                    if ybut.is_pressed:
                        y_in = True
                        break
                    ynum += 1
            if y_in and x_in:
                return f"{xnum},{ynum}"


def model_predict(model_int, input_array, verbose=False):
    # set up input
    input_details = model_int.get_input_details()[0]
    if verbose:
        print(input_array.dtype)
        print(input_array.shape)
        print(input_array)
        print(input_details)
    model_int.set_tensor(input_details['index'], input_array)

    # predict
    model_int.invoke()

    # get output
    output_details = model_int.get_output_details()[0]
    output = np.squeeze(model_int.get_tensor(output_details['index']))

    if verbose:
        print(output_details)
        print(output)
        model_vis1 = []
        largest_layer = 0
        for i in range(output_details['index'] + 1):
            layer = np.array(model_int.get_tensor(i)).ravel()
            if layer.dtype == 'float32' and layer.max() <= 1.0 and layer.min() >= -1.0:
                if layer.size > largest_layer:
                    largest_layer = layer.size
                # would like to map layer between 0 and 255 first
                # model_vis1.append(np.array(list(map(map_range, layer))))
                new_layer = []
                for item in layer:
                    if layer.size > 1:
                        new_layer.append(map_range(item, layer.min(), layer.max()))
                    else:
                        new_layer.append(map_range(item))
                model_vis1.append(np.array(new_layer))

                print(i)
                print(layer)
        print(largest_layer)            

        # print(model_vis1)

        # then iterate through model visual and pad every layer to the largest layer centering the actual values
        model_vis2 = []
        for mapped_layer in model_vis1:
            if mapped_layer.size < largest_layer:
                left_over = (largest_layer - mapped_layer.size) % 2
                left_side = int(((largest_layer - left_over) - mapped_layer.size) / 2)
                right_side = int(left_side + left_over)
                padded_layer = np.pad(mapped_layer, (left_side,right_side), constant_values=0)
                if padded_layer.size == largest_layer:
                    model_vis2.append(padded_layer)
                else:
                    print("layer NOT added! ", padded_layer.size)
            else:
                model_vis2.append(mapped_layer)
        model_vis3 = np.array(model_vis2)
        print(model_vis3)
        plt.imshow(model_vis3)
        plt.show()

    return output
    
        
class Player:
    def __init__(self, sign=1):
        self.sign = sign
        self.score = 0
        self.opponent_score = 0
        tflite_critic_model = 'save_models/citic_model.tflite'
        tflite_actor_model = 'save_models/actor_model.tflite'
        # Load the TFLite model in TFLite Interpreter
        self.critic_interpreter = tflite.Interpreter(tflite_critic_model)
        self.actor_interpreter = tflite.Interpreter(tflite_actor_model)

        self.critic_interpreter.allocate_tensors()
        self.actor_interpreter.allocate_tensors()
        

class BoardEnv:
    def __init__(self):
        self.p1 = Player(-1)
        # For actor
        self.round_buffer_O = []
        self.move_buffer_O = []
        self.round_buffer_X = []
        self.move_buffer_X = []
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
        self.previous_board_O = np.copy(self.board)
        self.previous_board_X = np.copy(self.board)
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
        self.ones = np.ones((3, 3))
        self.neg_ones = (np.ones((3, 3)) * -1)

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
        self.previous_board_O = np.copy(self.board)
        self.previous_board_X = np.copy(self.board)
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

    def make_move(self, x, y, just_data=False):
        if self.board[y][x] == 0:
            # for actor
            if not self.whose_turn:
                # for O's
                # for critic
                self.current_board = np.copy(self.board)
                random.seed(time.time_ns())
                value_calc_o = self.value_of_board(self.game_over, not self.whose_turn, symbol=-1)
                if value_calc_o != 0 or (random.randint(0, 100) == 50):
                    self.recorded_games.append([self.current_board, self.previous_board_O, self.neg_ones])
                    self.recorded_scores.append(value_calc_o)

                # for actor
                if not just_data:
                    move2 = np.copy(self.empty_board)
                    move2[y][x] = 1
                    if value_calc_o != 0:
                        value_of_board = model_predict(self.p1.critic_interpreter,
                                                               (np.array([[self.current_board.astype('float32'),
                                                                           self.previous_board_O.astype('float32'),
                                                                           self.neg_ones.astype('float32')]])), False)
                        value_layer = np.ones((3,3)) * value_of_board
                    else:
                        value_layer = np.zeros((3, 3))
                    self.move_buffer_O.append(move2)
                    self.round_buffer_O.append([self.current_board, self.previous_board_O, self.neg_ones, value_layer])

            else:
                # for X's
                # for critic
                self.current_board = np.copy(self.board)
                random.seed(time.time_ns())
                value_calc_x = self.value_of_board(self.game_over, self.whose_turn, symbol=1)
                if value_calc_x != 0 or (random.randint(0, 100) == 50):
                    self.recorded_games.append([self.current_board, self.previous_board_X, self.ones])
                    self.recorded_scores.append(value_calc_x)

                # for actor
                if not just_data:
                    move2 = np.copy(self.empty_board)
                    move2[y][x] = 1
                    if value_calc_x != 0:
                        value_of_board = model_predict(self.p1.critic_interpreter,
                                                               (np.array([[self.current_board.astype('float32'),
                                                                           self.previous_board_X.astype('float32'),
                                                                           self.ones.astype('float32')]])), False)
                        value_layer = (np.ones((3, 3)) * value_of_board)
                    else:
                        value_layer = np.zeros((3,3))
                    self.move_buffer_X.append(move2)
                    self.round_buffer_X.append([self.current_board, self.previous_board_X, self.ones, value_layer])

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
            go_led.on()
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
                if TERMINAL_INPUT:
                    move = input("what is your move? ")
                else:
                    move = visualize_board(self.board, False, human)
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
                            go_led.off()
                            time.sleep(1)
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
                        c_predict = model_predict(self.p1.critic_interpreter, (np.array([[self.current_board.astype('float32'),
                                                                            self.previous_board_O.astype('float32'),
                                                                            self.neg_ones.astype('float32')]])))
                        value_layer = (np.ones((3, 3)) * c_predict)
                        a_predict = model_predict(self.p1.actor_interpreter, (np.array([[self.current_board.astype('float32'),
                                                                           self.previous_board_O.astype('float32'),
                                                                           self.neg_ones.astype('float32'),
                                                                           value_layer.astype('float32')]])))
                        print(f"NN Value for O's: {c_predict}")
                        print(a_predict)
                    else:
                        c_predict = model_predict(self.p1.critic_interpreter, (np.array([[self.current_board.astype('float32'),
                                                                            self.previous_board_X.astype('float32'),
                                                                            self.ones.astype('float32')]])))
                        value_layer = (np.ones((3, 3)) * c_predict)
                        a_predict = model_predict(self.p1.actor_interpreter, (np.array([[self.current_board.astype('float32'),
                                                                           self.previous_board_X.astype('float32'),
                                                                           self.ones.astype('float32'),
                                                                           value_layer.astype('float32')]])))

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
                        c_predict = model_predict(self.p1.critic_interpreter, (np.array([[self.current_board.astype('float32'),
                                                                            self.previous_board_O.astype('float32'),
                                                                            self.neg_ones.astype('float32')]])), False)
                        value_layer = (np.ones((3, 3)) * c_predict)
                        pred = model_predict(self.p1.actor_interpreter, (np.array([[self.current_board.astype('float32'),
                                                                       self.previous_board_O.astype('float32'),
                                                                       self.neg_ones.astype('float32'),
                                                                       value_layer.astype('float32')]])), False)
                    else:
                        c_predict = model_predict(self.p1.critic_interpreter, (np.array([[self.current_board.astype('float32'),
                                                                            self.previous_board_X.astype('float32'),
                                                                            self.ones.astype('float32')]])), False)
                        value_layer = (np.ones((3, 3)) * c_predict)
                        pred = model_predict(self.p1.actor_interpreter, (np.array([[self.current_board.astype('float32'),
                                                                       self.previous_board_X.astype('float32'),
                                                                       self.ones.astype('float32'),
                                                                       value_layer.astype('float32')]])), False)
                    if verbose:
                        print('Algorithm playing!')
                        # print(pred)
                    if pred.any():
                        ai_y = 0
                        max = pred.ravel().max()
                        for row in pred:
                            row_max = int(np.argmax(row))
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
            if winner_ == -1 and value_board > 0.5:
                    self.actor_training.extend(self.round_buffer_O)
                    self.actor_moves.extend(self.move_buffer_O)
            elif winner_ == 1 and value_board > 0.5:
                self.actor_training.extend(self.round_buffer_X)
                self.actor_moves.extend(self.move_buffer_X)
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


# BUFFER ZONE

b1 = BoardEnv()
human_O = False
one_shot = False
one_shot_1000 = False
start_time = time.perf_counter()

while True:
    if b1.games_played % 5 == 0:
        if not one_shot:
            human_O = not human_O
        one_shot = True
    else:
        one_shot = False

    if human_O:
        b1.get_state(human=not b1.whose_turn, random_play=False, verbose=True)

    else:
        b1.get_state(human=b1.whose_turn, random_play=False, verbose=True)

