import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

# Set matplotlib color map to greyscale always
plt.rc('image', cmap='gray')

# Read datasets
fea = np.array(scipy.io.loadmat('../resources/ORL_32x32.mat')['fea'])
gnd = np.array(scipy.io.loadmat('../resources/ORL_32x32.mat')['gnd'])
train_3 = np.array(scipy.io.loadmat('../resources/3Train/3.mat'))
train_5 = np.array(scipy.io.loadmat('../resources/5Train/5.mat'))
train_7 = np.array(scipy.io.loadmat('../resources/7Train/7.mat'))


def print_faces(number_of_images, images_per_row):
    images_per_column = math.ceil(number_of_images / images_per_row)
    shape = (32 * images_per_column, 32 * images_per_row)
    print_array = np.zeros(shape)

    row = 0
    col = 0
    for i in range(number_of_images):
        if i % images_per_row == 0 and i != 0:
            row += 32
            col = 0
        print_array[row:row + 32, col:col + 32] = fea[i].reshape(32, 32).T
        col += 32

    plt.imshow(print_array)
    plt.show()

def print_face(face):
    plt.imshow(face.reshape(32, 32).T)
    plt.show()


if __name__ == "__main__":
    # Print 100 faces
    print_faces(100, 10)

    # Print mean face of everyone
    print_face(fea.mean(axis=0))


