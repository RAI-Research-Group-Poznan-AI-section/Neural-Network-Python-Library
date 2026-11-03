import main_class as nn

neural_network = nn.polska_gurom([784, 30, 10]) 
data = nn.load_data(".\\pictures", ".\\labels.csv")
test = nn.load_data(".\\white_test_pictures", ".\\white_test_labels.csv")
neural_network.train(data, epochs=30, batch_size=10, learning_rate=3.0, test_data=test)
