import  board
import neopixel
import time
import socket

pixel = neopixel.NeoPixel(board.D18, 9, pixel_order=neopixel.GRB)
X_COLOR = (255,0,0)
O_COLOR = (0,0,255)

host = socket.gethostname()  # get local machine name
port = 8080  # Make sure it's within the > 1024 $$ <65535 range

s = socket.socket()
s.bind((host, port))

s.listen(1)
client_socket, adress = s.accept()
print("Connection from: " + str(adress))

while True:
    data = s.recv(1024)
    if not data:
        continue
    print('From online user: ' + data)
    for box in range(9):
        pix = box
        if box == 3:
            pix = 5
        elif box == 5:
            pix = 3

        elif data[box] == -1:
            pixel[pix] = X_COLOR
        elif data[box] == 1:
            pixel[pix] = O_COLOR
        else:
            pixel[pix] = (0, 0, 0)

    pixel.show()
    data = 'Pixels shown'
    s.send(data.encode('utf-8'))
s.close()



