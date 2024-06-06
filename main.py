import numpy as np
import pandas as pd
from numpy import sign
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

NUM_OF_DATA_IN_CIRCLE_1 = 300
LENGTH_LOSSES_GRAPH = 350
def perceptron_train_for_one_digit(X_train, y_train, i):
    """
    Trains a perceptron model for a specific digit.

    Parameters:
    - X_train: Training data features.
    - y_train: Training data labels.
    - i: Digit for which the perceptron is trained.

    Returns:
    - w: Final weight matrix for the digit.
    """
    numSamples = len(X_train)
    w = np.zeros((LENGTH_LOSSES_GRAPH, X_train.shape[1]))
    wTemp = np.full(X_train[1].shape, 0.0)
    y_sample = np.where(y_train == i, 1, -1)

    ind = 0
    j = 0
    max_matches = 0
    num_circles = LENGTH_LOSSES_GRAPH - NUM_OF_DATA_IN_CIRCLE_1
    frequency_of_testing = len(X_train) / NUM_OF_DATA_IN_CIRCLE_1
    circle = 0

    while circle < num_circles:
        y = sign(np.dot(wTemp, X_train[ind]))

        # Check for misclassification
        if y != y_sample[ind]:
            # Update weights using perceptron learning rule
            wTemp += X_train[ind] * y_sample[ind]
            j += 1
            # Periodically check for matches and update wMax if necessary
            if j % 10 == 0:
                y_temp = np.dot(wTemp, X_train[ind])
                matches = np.sum(np.where(y_temp > 0, 1, -1) == y_sample)
                if matches > max_matches:
                    max_matches = matches
                    wMax = wTemp
        ind += 1
        # Collecting information for the presentation from the first cycle of the perceptron
        if (circle == 0) & (ind % frequency_of_testing == 0):
            w[int(ind / frequency_of_testing)] = wMax

        # Reset index when all samples are processed in a cycle
        if ind == numSamples:
            ind = 0
            # Save wMax at the end of each cycle
            if circle < LENGTH_LOSSES_GRAPH - NUM_OF_DATA_IN_CIRCLE_1:
                w[circle + NUM_OF_DATA_IN_CIRCLE_1] = wMax

            circle += 1
    return w


def calculate_accuracy_over_iterations(X, y, w, LENGTH_LOSSES_GRAPH):
    """
    Calculates accuracy over iterations for the perceptron model.

    Parameters:
    - X: Data features for which accuracy is calculated.
    - y: True labels for the data.
    - w: Weight matrix for the perceptron.
    - LENGTH_LOSSES_GRAPH: Number of iterations.

    Returns:
    - accuracy_over_iterations: Array of accuracy values over iterations.
    """
    accuracy_over_iterations = np.zeros(LENGTH_LOSSES_GRAPH)

    for h in range(LENGTH_LOSSES_GRAPH):
        # Make predictions and find the index of the maximum value (digit)
        predictions = np.argmax(np.dot(w[:, h, :], X.T), axis=0)
        # Count correct predictions
        correct_predictions = np.sum(predictions == y)
        # Calculate the percentage of error
        accuracy_over_iterations[h] = 1 - correct_predictions / len(X)
    return accuracy_over_iterations

def perceptron_test(X_test, y_test, w):
    """
    Tests the perceptron model on the test data and provides detailed evaluation metrics.

    Parameters:
    - X_test: Test data features.
    - y_test: True labels for the test data.
    - w: Weight matrix for the perceptron.
    """
    evaluate = {i: np.zeros(7) for i in range(10)}
    predictions = np.argmax(np.dot(w[:, LENGTH_LOSSES_GRAPH - 1, :], X_test.T), axis=0)

    # Evaluate predictions
    for i in range(len(y_test)):
        if np.equal(predictions[i], y_test[i]):
            # Updating the positive columns
            evaluate[y_test[i]][0] += 1
            for j in range(10):
                if j != y_test[i]:
                    evaluate[j][1] += 1
        else:
            # Updating the negative columns
            evaluate[predictions[i]][2] += 1
            for j in range(10):
                if j != y_test[i] and j != predictions[i]:
                    evaluate[j][1] += 1
                else:
                    if j == y_test[i]:
                        evaluate[j][3] += 1

    # Calculate additional evaluation metrics
    for i in range(10):
        evaluate[i][4] = float(evaluate[i][0] + evaluate[i][1]) / float(
            evaluate[i][0] + evaluate[i][1] + evaluate[i][2] + evaluate[i][3])
        evaluate[i][5] = float(evaluate[i][0]) / float(evaluate[i][0] + evaluate[i][3])
        evaluate[i][6] = float(evaluate[i][1]) / float(evaluate[i][1] + evaluate[i][2])

    df = pd.DataFrame.from_dict(evaluate, orient='index', columns=['TP', 'TN', 'FP', 'FN', 'ACC', 'TPR', 'TNR'])
    print(df)

    # Visualize confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def train_and_test_losses_graph(X_train, y_train, X_test, y_test, w):
    """
    Plots the training and testing losses over iterations.

    Parameters:
    - X_train, y_train: Training data features and labels.
    - X_test, y_test: Test data features and labels.
    - w: Weight matrix for the perceptron.
    - LENGTH_LOSSES_GRAPH: Number of iterations.
    """
    train_losses = calculate_accuracy_over_iterations(X_train, y_train, w, LENGTH_LOSSES_GRAPH)
    test_losses = calculate_accuracy_over_iterations(X_test, y_test, w, LENGTH_LOSSES_GRAPH)

    # Plot the accuracy and error
    plt.scatter(range(len(train_losses)), train_losses, c="red", label="Train")
    plt.scatter(range(len(test_losses)), test_losses, c="blue", label="Test")

    # Add a legend
    plt.legend()
    cycle_1_end = NUM_OF_DATA_IN_CIRCLE_1 - 1

    # Mark the end of the first cycle
    plt.axvline(x=cycle_1_end, color='green', linestyle='--', label='End of Cycle 1')
    plt.text(cycle_1_end + 0.5, plt.ylim()[1] - 0.27, 'End of Cycle 1', color='green', rotation=90,
             verticalalignment='bottom')
    plt.show()


def perceptron(X_train, y_train, X_test, y_test):
    """
    Main function orchestrating the entire perceptron training and testing pipeline.

    Parameters:
    - X_train, y_train: Training data features and labels.
    - X_test, y_test: Test data features and labels.
    """

    # Initialize the weight matrix
    w = np.zeros((10, LENGTH_LOSSES_GRAPH, X_train.shape[1]))

    # Train the perceptron for each digit
    for i in range(10):
        w[i] = perceptron_train_for_one_digit(X_train, y_train, i)

    # Test the perceptron on the test set
    perceptron_test(X_test, y_test, w)
    # Plot the training and testing losses
    train_and_test_losses_graph(X_train, y_train, X_test, y_test, w)

if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    # Fetch MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1 / 7, random_state=42)

    # Add bias terms to feature matrices
    X_train_with_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_test_with_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])

    # Convert labels to integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Train and test the perceptron
    perceptron(X_train_with_bias, y_train, X_test_with_bias, y_test)
