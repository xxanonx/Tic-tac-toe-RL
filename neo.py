import  board
import neopixel
import time
import socket
import pickle


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1234))

pixel = neopixel.NeoPixel(board.D18, 9)
X_COLOR = (255,0,0)
O_COLOR = (0,0,255)

pixel.fill((0,0,0))
for i in range(9):
    pixel[i] = (255,0,255)
    pixel.show()
    time.sleep(1)
pixel.fill((0,0,0))

while True:
    msg = s.recv(1024)
    if len(msg) > 0:
        print(msg)
        print(len(msg))
        board = pickle.loads(msg)
        print(board)
    
        for box in range(len(board)):
            pix = box
            if box == 3:
                pix = 5
            elif box == 5:
                pix = 3

            elif board[box] == -1:
                pixel[pix] = X_COLOR
            elif board[box] == 1:
                pixel[pix] = O_COLOR
            else:
                pixel[pix] = (0, 0, 0)
        pixel.show()



