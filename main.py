import numpy as np

test_dataset_inputs = np.array(([1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 0, 0, 0, 0],
                                [1, 1, 0, 1, 1, 0, 1],
                                [1, 1, 1, 1, 0, 0, 1],
                                [0, 1, 1, 0, 0, 1, 1],
                                [1, 0, 1, 1, 0, 1, 1],
                                [1, 0, 1, 1, 1, 1, 1],
                                [1, 1, 1, 0, 0, 0, 0],
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 0, 1, 1],
                                [1, 1, 1, 0, 1, 1, 1],
                                [0, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 0, 1],
                                [1, 0, 0, 1, 1, 1, 1],
                                [1, 0, 0, 0, 1, 1, 1]), dtype=float)

test_dataset_expected_results = np.array(([0, 0, 0, 0],
                                          [0, 0, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 1, 1],
                                          [0, 1, 0, 0],
                                          [0, 1, 0, 1],
                                          [0, 1, 1, 0],
                                          [0, 1, 1, 1],
                                          [1, 0, 0, 0],
                                          [1, 0, 0, 1],
                                          [1, 0, 1, 0],
                                          [1, 0, 1, 1],
                                          [1, 1, 0, 0],
                                          [1, 1, 0, 1],
                                          [1, 1, 1, 0],
                                          [1, 1, 1, 1]), dtype=float)


class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers_count = len(layers)
        assert self.layers_count >= 3  # at least one hidden layer to crete neural network

        self.layers_sizes = layers
        # parameters
        self.weighted_inputs = []

        self.w = []
        for index in range(self.layers_count - 1):
            self.w.append(np.random.randn(self.layers_sizes[index], self.layers_sizes[index + 1]))

    def feed_forward(self, training_data):
        self.weighted_inputs = []
        layer_input = training_data
        for index in range(self.layers_count - 1):
            weighted_inputs = np.dot(layer_input, self.w[index])
            self.weighted_inputs.append(self.sigmoid(weighted_inputs))
            layer_input = self.weighted_inputs[index]
        return layer_input

    @staticmethod
    def sigmoid(s):
        return 1 / (1 + np.exp(-s))

    @staticmethod
    def sigmoid_derivative(s):
        return s * (1 - s)

    def backward(self, testing_data, expected_results, output):
        error = expected_results - output
        for layer_index in reversed(range(self.layers_count - 1)):
            delta = error * self.sigmoid_derivative(output)
            if layer_index <= 0:
                input_from_prev_layer = testing_data
            else:
                input_from_prev_layer = self.weighted_inputs[layer_index - 1]
                error = delta.dot(self.w[layer_index].T)
                output = self.weighted_inputs[layer_index - 1]
            self.w[layer_index] += input_from_prev_layer.T.dot(delta)

    def train(self, training_data, expected_results):
        # training data rows count must be equal to the number of neural network inputs
        assert training_data.shape[1] == self.layers_sizes[0]
        # training data results rows count must be equal to the number of neural network outputs
        assert expected_results.shape[1] == self.layers_sizes[self.layers_count - 1]

        output = self.feed_forward(training_data)
        self.backward(training_data, expected_results, output)


if __name__ == '__main__':
    neural_network = NeuralNetwork([7, 9, 9, 10, 15, 4])

    for i in range(1000):  # trains neural network 1000 times
        if i % 100 == 0:
            print("Loss: " + str(
                np.mean(np.square(test_dataset_expected_results - neural_network.feed_forward(test_dataset_inputs)))))
        neural_network.train(test_dataset_inputs, test_dataset_expected_results)

    print("Input: " + str(test_dataset_inputs))
    print("Predicted Output: " + str(neural_network.feed_forward(test_dataset_inputs)))
    print("Expected Output: " + str(test_dataset_expected_results))
    print("Loss: " + str(np.mean(np.square(test_dataset_expected_results -
                                           neural_network.feed_forward(test_dataset_inputs)))))
    print("\n")
