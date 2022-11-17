import numpy as np

learning_dataset_inputs = np.array(([1, 1, 1, 1, 1, 1, 0],
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

learning_dataset_expected_results = np.array(([0, 0, 0, 0],
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

testing_dataset_inputs = learning_dataset_inputs
testing_dataset_expected_results = learning_dataset_expected_results


class NeuralNetwork(object):
    def __init__(self, layers_sizes):
        self.layers_count = len(layers_sizes)
        assert self.layers_count >= 3  # at least one hidden layer to crete neural network
        self.layers_sizes = layers_sizes
        self.weighted_inputs = None
        self.weights = self.generate_synapses_weights(self.layers_count)

    def generate_synapses_weights(self, layers_count):
        weights = []
        for index in range(layers_count - 1):
            weights.append(np.random.randn(self.layers_sizes[index], self.layers_sizes[index + 1]))
        return weights

    def train(self, training_data, expected_results):
        # training data rows count must be equal to the number of neural network inputs
        assert training_data.shape[1] == self.layers_sizes[0]
        # training data results rows count must be equal to the number of neural network outputs
        assert expected_results.shape[1] == self.layers_sizes[self.layers_count - 1]

        output = self.feed_forward(training_data)
        self.backward_propagate_learning(training_data, expected_results, output)

    def feed_forward(self, training_data):
        self.weighted_inputs = []
        layer_input = training_data
        for index in range(self.layers_count - 1):
            weighted_inputs = np.dot(layer_input, self.weights[index])
            self.weighted_inputs.append(self.sigmoid(weighted_inputs))
            layer_input = self.weighted_inputs[index]
        return layer_input

    def backward_propagate_learning(self, testing_data, expected_results, output):
        layer_output = output
        error = expected_results - output

        for layer_index in reversed(range(1, self.layers_count - 1)):  # adjust hidden layers
            delta_adjustment = error * self.sigmoid_derivative(layer_output)  # compute weight adjustment
            error = delta_adjustment.dot(self.weights[layer_index].T)
            layer_output = self.weighted_inputs[layer_index - 1]
            self.weights[layer_index] += layer_output.T.dot(delta_adjustment)

        # adjust input layer
        self.weights[0] += testing_data.T.dot(error * self.sigmoid_derivative(layer_output))

    @staticmethod
    def sigmoid(s):
        return 1 / (1 + np.exp(-s))

    @staticmethod
    def sigmoid_derivative(s):
        return s * (1 - s)


def learn(testing_data, testing_data_results):
    assert testing_data.shape[0] == testing_data_results.shape[0]
    inputs_count = testing_data.shape[1]
    outputs_count = testing_data_results.shape[1]
    neural_network = NeuralNetwork([inputs_count, 8, 6, 7, outputs_count])
    print("====== LEARNING: ======")
    for i in range(10_000):  # trains neural network 1000 times
        if i % 1000 == 0:
            loss = np.mean(
                np.square(testing_data_results - neural_network.feed_forward(testing_data)))
            print(
                f"Success ratio: {str(100 - loss * 100)} %")
        neural_network.train(testing_data, testing_data_results)
    print("\n")

    print("====== TESTING: ======")
    feed_forward = neural_network.feed_forward(testing_dataset_inputs)
    loss = np.mean(np.square(testing_data_results - feed_forward))
    print("Input: " + str(testing_dataset_inputs))
    print("Predicted Output: " + str(feed_forward))
    print("Expected Output: " + str(testing_data_results))
    print(f"Success ratio: {str(100 - loss * 100)} %")
    print("\n")


if __name__ == '__main__':
    learn(learning_dataset_inputs, learning_dataset_expected_results)
