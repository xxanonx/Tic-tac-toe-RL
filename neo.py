import  board
import neopixel
import time
import socket
import os

pixel = neopixel.NeoPixel(board.D18, 9, pixel_order=neopixel.GRB)
X_COLOR = (255,0,0)
O_COLOR = (0,0,255)
server = "/tmp/socket_test.s"

'''pixel.fill((0,0,0))
for i in range(9):
    pixel[i] = (255,0,255)
    pixel.show()
    time.sleep(1)
pixel.fill((0,0,0))'''

if os.path.exists(server):
    os.unlink(server)

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(server)

while True:
    s.listen(1)
    connection, adress = s.accept()
    print("Connection from: " + str(adress))
    data = connection.recv(1024)
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
    connection.send(data.encode('utf-8'))
s.close()



