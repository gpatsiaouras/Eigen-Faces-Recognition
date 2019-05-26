import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
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


def print_faces(data, number_of_images, images_per_row):
    images_per_column = math.ceil(number_of_images / images_per_row)
    shape = (32 * images_per_column, 32 * images_per_row)
    print_array = np.zeros(shape)

    row = 0
    col = 0
    for i in range(number_of_images):
        if i % images_per_row == 0 and i != 0:
            row += 32
            col = 0
        print_array[row:row + 32, col:col + 32] = data[i].reshape(32, 32).T
        col += 32

    plt.imshow(print_array)
    plt.colorbar()
    plt.show()


def print_face(face, title):
    plt.title(title)
    plt.imshow(face.reshape(32, 32).T)
    plt.colorbar()
    plt.show()


def pca(k):
    standardized_fea = fea / 255

    # Calculate the mean of each column
    mean = standardized_fea.mean(axis=0)

    # Subtract mean from columns
    subtracted = standardized_fea - mean

    # Calculate covariance matrix
    covariance = np.cov(subtracted.T)

    # Eigendecomposition of covariance
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort by highest principal component.
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    # print_faces(eigenvectors.T, number_of_images=30, images_per_row=6)

    # Reduce dimensionality
    eigenvectors = eigenvectors.T[:k]

    # Project data
    projected = subtracted.dot(eigenvectors.T)

    return projected, eigenvectors, mean


if __name__ == "__main__":
    k = 10

    # Apply pca
    projected, eigen_vectors, mean = pca(k)

    # Plot Mean face
    print_face(mean, "mean")

    # Plot first 30 components
    print_faces(eigen_vectors, number_of_images=10, images_per_row=5)

    # Reconstruct first image
    reconstructed = mean + projected[0, :].reshape(1, k).dot(eigen_vectors)
    print_face(reconstructed, "Reconstructed")
