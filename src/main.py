import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
from scipy.spatial.distance import cdist
import math
from tabulate import tabulate

# Set matplotlib color map to greyscale always
plt.rc('image', cmap='gray')

# Read datasets from matlab files
ORL = scipy.io.loadmat('../resources/ORL_32x32.mat')
data_3 = scipy.io.loadmat('../resources/3Train/3.mat')
data_5 = scipy.io.loadmat('../resources/5Train/5.mat')
data_7 = scipy.io.loadmat('../resources/7Train/7.mat')

# Assign to numpy variables
full_data = np.array(ORL['fea'])
labels = np.array(ORL['gnd'])
train_3 = np.array(data_3['trainIdx'])
test_3 = np.array(data_3['testIdx'])
train_5 = np.array(data_5['trainIdx'])
test_5 = np.array(data_5['testIdx'])
train_7 = np.array(data_7['trainIdx'])
test_7 = np.array(data_7['testIdx'])


def print_multiple_faces(images, titles, rows, columns, should_save=True):
    """
    This function is used to print a lot of faces in one plot in an easy and automated way.
    :param images: Vectors of images
    :param titles: Titles for each of the images
    :param rows: Rows that the subplots should have
    :param columns: Columns that the subplots should have
    :param should_save: Save to folder or not
    """
    figure = plt.figure(1)
    for idx in range(len(images)):
        figure.add_subplot(rows, columns, idx + 1)
        plt.title(titles[idx])
        plt.axis('off')
        plt.imshow(images[idx].reshape(32, 32).T)

    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    if should_save:
        plt.savefig("../resources/output/{0}_multifigure.jpg".format(time.time()))
    plt.show()


def print_faces(data, title, number_of_images, images_per_row, should_save=True):
    """
    Prints the vectors of images contained in a matrix.
    :param data: Matrix containing vectors of images
    :param title: Title
    :param number_of_images: number of images to print in total
    :param images_per_row: Images per row
    :param should_save: Save to folder
    """
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
    plt.axis('off')
    plt.imshow(print_array)
    # plt.colorbar()
    if should_save:
        plt.savefig("../resources/output/{0}_multifaces.jpg".format(time.time()))
    plt.show()


def print_face(face, title, should_save=True):
    """
    Print one face from vector
    :param face: Face vector
    :param title: Title
    :param should_save: Save to folder
    """
    plt.title(title)
    plt.imshow(face.reshape(32, 32).T)
    plt.colorbar()
    plt.axis('off')
    if should_save:
        plt.savefig("../resources/output/{0}_face.jpg".format(time.time()))
    plt.show()


def plot_accuracies(x_axis_data, data, titles):
    """
    Plot fancy graph of the accuracy
    :param x_axis_data: k values
    :param data: accuracy per data set
    :param titles: Titles for legend
    """
    # Plot each content of the data list
    for i in range(len(data)):
        plt.plot(x_axis_data, data[i], label=titles[i])
        plt.ylabel('Accuracy')
        plt.xlabel('k Value')
        plt.legend()

    plt.show()


def print_pretty_accuracies(k_range, accuracies_for_train_3, accuracies_for_train_5, accuracies_for_train_7):
    """
    Print a fancy table for readability of the values
    :param k_range: K values
    :param accuracies_for_train_3: accuracy list for train 3
    :param accuracies_for_train_5: accuracy list for train 5
    :param accuracies_for_train_7: accuracy list for train 7
    """
    accuracies_for_train_3.insert(0, "Accuracy of 3")
    accuracies_for_train_5.insert(0, "Accuracy of 5")
    accuracies_for_train_7.insert(0, "Accuracy of 7")
    print(tabulate([accuracies_for_train_3, accuracies_for_train_5, accuracies_for_train_7], headers=k_range))


def pca(data, k):
    """
    Principal Component Analysis
    :param data: data
    :param k: number of features to keep
    :return: Projected values to the space, eigenvectors, mean
    """
    # Calculate the mean of each column
    mean = data.mean(axis=0)

    # Subtract mean from columns
    subtracted = data - mean

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


def get_nearest_neighbor(training_set, test_set):
    """
    Nearest neighbor algorithm, calculates distances between training_set and test_set, then sorts the distances and
    takes the closest id of training_Set for each of the elements of test_set. Returns a vector
    :param training_set:
    :param test_set:
    :return: Vector of indices
    """
    # Calculate distances
    distances = cdist(training_set, test_set)

    # Sort distances and get back the indices of the sorted list
    sorted_indices = np.argmin(distances, axis=0)

    # Get first neighbor
    return sorted_indices


def reconstruct_image_from_eigenvectors(image_id):
    """
    Runs pca with 10,20,30 k values
    :param image_id:
    :return:
    """
    projected_10, eigen_vectors_10, mean = pca(full_data, 10)
    projected_20, eigen_vectors_20, mean = pca(full_data, 20)
    projected_30, eigen_vectors_30, mean = pca(full_data, 30)

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


def get_accuracy(test_indices, train_indices, predictions_idx):
    """
    Find the identities of the test subjects, find the identities of the predicted subjects from the training
    set and compare the two arrays.
    :param test_indices:
    :param train_indices:
    :param predictions_idx:
    :return: accuracy percentage
    """
    identities_of_test_subjects = labels[test_indices]
    identities_of_predicted_people = labels[train_indices[predictions_idx]]

    return np.count_nonzero(identities_of_test_subjects == identities_of_predicted_people) / \
           len(identities_of_test_subjects)


def face_recognition(train, test, k, should_print_differences=True):
    """
    This function performs pca on training and test set and projects the data in the
    new features space. Then it uses the Nearest Neighbor algorithm to fetch the closest
    face to the test data.
    """
    # Take train 3 data (indexes -1 for matlab compatibility)
    train_indices = train.flatten() - 1
    test_indices = test.flatten() - 1

    # Extract actual data based on indices
    training_data = full_data[train_indices]
    test_data = full_data[test_indices]

    # Standardize data
    training_data = training_data / 255
    test_data = test_data / 255

    # Apply pca with different k values on training data
    projected_train, eigen_vectors_train, mean_train = pca(training_data, k)

    # Subtract the mean from test_data
    test_data = test_data - mean_train

    # Project test data on the space of train data.
    projected_test = test_data.dot(eigen_vectors_train.T)

    # Find the nearest faces for all the faces in the test dataset
    predictions_idx = get_nearest_neighbor(projected_train, projected_test)

    # Calculate accuracy
    accuracy = get_accuracy(test_indices, train_indices, predictions_idx)

    # Plot the 8 first entries of the test data and the corresponding predicted ones
    if should_print_differences:
        print_multiple_faces(
            [
                full_data[test_indices[0]], full_data[train_indices[predictions_idx[0]]],
                full_data[test_indices[1]], full_data[train_indices[predictions_idx[1]]],
                full_data[test_indices[2]], full_data[train_indices[predictions_idx[2]]],
                full_data[test_indices[3]], full_data[train_indices[predictions_idx[3]]],
                full_data[test_indices[4]], full_data[train_indices[predictions_idx[4]]],
                full_data[test_indices[5]], full_data[train_indices[predictions_idx[5]]],
                full_data[test_indices[6]], full_data[train_indices[predictions_idx[6]]],
                full_data[test_indices[7]], full_data[train_indices[predictions_idx[7]]]
            ],
            [
                "Original 0", "Predicted 0",
                "Original 1", "Predicted 1",
                "Original 2", "Predicted 2",
                "Original 3", "Predicted 3",
                "Original 4", "Predicted 4",
                "Original 5", "Predicted 5",
                "Original 6", "Predicted 6",
                "Original 7", "Predicted 7"
            ],
            4, 4
        )

    return accuracy


def experiment_k_values_and_datasets_face_recognition():
    """
    Make an experiment and try the accuracy of each of the train3, 5 and 7 datasets with different
    k values
    """
    # Initiate lists to store the accuracies for each k value
    accuracies_for_train_3 = []
    accuracies_for_train_5 = []
    accuracies_for_train_7 = []

    # Select a range of k values
    k_range = range(2, 200, 10)

    # Try all 3 trainsets for each k value
    for k in k_range:
        accuracies_for_train_3.append(face_recognition(train_3, test_3, k, False))
        accuracies_for_train_5.append(face_recognition(train_5, test_5, k, False))
        accuracies_for_train_7.append(face_recognition(train_7, test_7, k, False))

    plot_accuracies(
        k_range,
        [accuracies_for_train_3, accuracies_for_train_5, accuracies_for_train_7],
        ["Accuracies for train 3", "Accuracies for train 5", "Accuracies for train 7"]
    )

    print_pretty_accuracies(k_range, accuracies_for_train_3, accuracies_for_train_5, accuracies_for_train_7)


if __name__ == "__main__":
    # Plot first 16 components of pca
    _, eigen_vectors_200, _ = pca(full_data, 200)
    print_faces(eigen_vectors_200, "Eigen vectors", number_of_images=25, images_per_row=5)

    # Reconstruct image from eigen vectors
    reconstruct_image_from_eigenvectors(image_id=0)

    # Perform face recognition using NN algorithm and train 3 dataset
    accuracy = face_recognition(train_3, test_3, k=30)
    print("Accuracy: {0:.2f} %".format(accuracy * 100))

    # Perform experiment with different k values and datasets, takes a bit of time
    experiment_k_values_and_datasets_face_recognition()
