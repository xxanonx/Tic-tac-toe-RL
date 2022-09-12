import numpy as np
import matplotlib.pyplot as plt


def visualize(board):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for color, add in [('b', -1), ('r', 1), ('0.8', 0)]:
        x = []
        y = []
        z = []
        layer_num = 0
        temp_board = np.copy(board)
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


# array3d = np.zeros((3, 3, 3))
# array2d = np.ones((3, 3))

# array3d[0] = np.copy(array2d)
# array3d[1] = np.copy(array2d * -1)
array3d = np.array(
        [[[-1, -1, -1],
        [-1, 1, 1],
        [1, 0, 0]],
       [[-1, 0, -1],
        [1, -1, 1],
        [0, 0, 0]],
       [[1, 0, -1],
        [0, 0, -1],
        [0, 0, 0]]])
print(array3d)

visualize(array3d)
visualize(np.moveaxis(array3d, 0, -1))
visualize(np.moveaxis(array3d, -1, 0))



