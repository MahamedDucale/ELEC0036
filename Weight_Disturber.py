from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the model.
model = Sequential([
  Dense(100, activation='relu', input_shape=(784,)),
  Dense(10, activation='softmax'),
])

# Load the model's saved weights.
model.load_weights('Relu_files/Relu_model_1.h5')

# Select weights for layers
weights = model.get_weights()
# Weights for hidden layer
layer_1_weights = weights[0]
# Weights for output layer
layer_2_weights = weights[2]

