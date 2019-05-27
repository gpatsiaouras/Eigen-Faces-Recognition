import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
from scipy.spatial.distance import cdist
import math

# Set matplotlib color map to greyscale always
plt.rc('image', cmap='gray')

# Read datasets from matlab files
ORL = scipy.io.loadmat('../resources/ORL_32x32.mat')
data_3 = scipy.io.loadmat('../resources/3Train/3.mat')
data_5 = scipy.io.loadmat('../resources/5Train/5.mat')
data_7 = scipy.io.loadmat('../resources/7Train/7.mat')

# Assign to numpy variables
fea = np.array(ORL['fea'])
gnd = np.array(ORL['gnd'])
train_3 = np.array(data_3['trainIdx'])
test_3 = np.array(data_3['testIdx'])
train_5 = np.array(data_5['trainIdx'])
test_5 = np.array(data_5['testIdx'])
train_7 = np.array(data_7['trainIdx'])
test_7 = np.array(data_7['testIdx'])


def print_multiple_faces(images, titles, rows, columns, should_save=True):
    figure = plt.figure(1)
    for idx in range(len(images)):
        figure.add_subplot(rows, columns, idx + 1)
        plt.title(titles[idx])
        plt.imshow(images[idx].reshape(32, 32).T)

    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    if should_save:
        plt.savefig("../resources/output/{0}_multifigure.jpg".format(time.time()))
    plt.show()



def print_faces(data, title, number_of_images, images_per_row, should_save=True):
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

    plt.title(title)
    plt.imshow(print_array)
    # plt.colorbar()
    if should_save:
        plt.savefig("../resources/output/{0}_multifaces.jpg".format(time.time()))
    plt.show()


def print_face(face, title, should_save=True):
    plt.title(title)
    plt.imshow(face.reshape(32, 32).T)
    plt.colorbar()
    if should_save:
        plt.savefig("../resources/output/{0}_face.jpg".format(time.time()))
    plt.show()


def pca(data, k):
    standardized_data = data / 255

    # Calculate the mean of each column
    mean = standardized_data.mean(axis=0)

    # Subtract mean from columns
    subtracted = standardized_data - mean

    # Calculate covariance matrix
    covariance = np.cov(subtracted.T)

    # Eigendecomposition of covariance
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort by highest principal component.
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Reduce dimensionality by taking the first k
    eigenvectors = eigenvectors.T[:k]

    # Project data
    projected = subtracted.dot(eigenvectors.T)

    return projected, eigenvectors, mean


def get_nearest_neighbors(training_set, test_element):
    distances = cdist(training_set, test_element, metric='euclidean')
    return None


def reconstruct_image_from_eigenvectors(image_id):
    # Reconstruct first image
    reconstructed = [
        mean,
        mean + projected_10[image_id, :].reshape(1, 10).dot(eigen_vectors_10),
        mean + projected_20[image_id, :].reshape(1, 20).dot(eigen_vectors_20),
        mean + projected_30[image_id, :].reshape(1, 30).dot(eigen_vectors_30)
    ]

    titles = [
        "Mean face",
        "Reconstructed from k=10",
        "Reconstructed from k=20",
        "Reconstructed from k=30"
    ]

    print_multiple_faces(reconstructed, titles, 2, 2)


if __name__ == "__main__":
    # Take train 3 data (indexes -1 for matlab compatibility)
    training_data = fea[train_3.flatten() - 1]
    test_data = fea[test_3.flatten() - 1]

    # Apply pca with different k values on training data
    projected_10, eigen_vectors_10, mean = pca(training_data, 10)
    projected_20, eigen_vectors_20, _ = pca(training_data, 20)
    projected_30, eigen_vectors_30, _ = pca(training_data, 30)

    # Apply pca with different k values on training data
    projected_10_test, eigen_vectors_10_test, mean_test = pca(test_data, 10)
    projected_20_test, eigen_vectors_20_test, _ = pca(test_data, 20)
    projected_30_test, eigen_vectors_30_test, _ = pca(test_data, 30)

    # Plot first 16 components of pca
    _, eigen_vectors_200, _ = pca(training_data, 200)
    print_faces(eigen_vectors_30_test, "Eigen vectors", number_of_images=16, images_per_row=4)

    # Reconstruct image from eigen vectors
    reconstruct_image_from_eigenvectors(image_id=0)
