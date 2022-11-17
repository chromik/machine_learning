import numpy as np


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


def create_testing_data():
    # generate 100 x 5-bit (5 inputs with 0 or 1 value) created 100x with random value
    testing_data = np.random.randint(0, 2, (200, 5))
    results = []
    for row in testing_data:
        # converts 5-bit binary value into decimal format abd scale
        results.append((row[0] * 16 + row[1] * 8 + row[2] * 4 + row[3] * 2 + row[4]) / 31)
    testing_data_results = np.array([results]).T  # transpose matrix to have each result on the new row
    return testing_data, testing_data_results


def learn(network, iterations=1_000):
    print("====== LEARNING: ======")
    for i in range(iterations):
        learning_data, learning_data_results = create_testing_data()

        if i % 1000 == 0:
            loss = np.mean(np.square(learning_data_results - network.feed_forward(learning_data)))
            print(f"Success ratio: {str(100 - (loss * 31) * 100)} %  ({i}/{iterations})")
            print("\n")

        network.train(learning_data, learning_data_results)
    print("\n")


def test(network):
    print("====== TESTING: ======")
    testing_data, testing_data_results = create_testing_data()
    raw_output = network.feed_forward(testing_data)
    loss = np.mean(np.square(testing_data_results - raw_output))
    print("Input: " + str(testing_data))
    print("Predicted Output: " + str(raw_output * 31))  # scale output back
    print("Expected Output: " + str(testing_data_results * 31))  # scale output back
    print(f"Success ratio: {str(100 - (loss * 31) * 100)} %")
    print("\n")


if __name__ == '__main__':
    neural_network = NeuralNetwork([5, 8, 6, 1])
    learn(neural_network, 200_000)
    test(neural_network)
