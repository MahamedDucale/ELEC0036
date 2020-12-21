from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import badmemristor
import badmemristor.nonideality
import numpy as np
import mnist
import csv

def max_and_min_weights(arrayofweights,HRSLRS,Pl):
  flatarray = arrayofweights.flatten()
  absarray = np.absolute(flatarray)
  ascsortedarray = np.sort(absarray)
  descsortedarray = ascsortedarray[::-1]
  size = len(descsortedarray)
  index = int(Pl * size)
  Wmax = descsortedarray[index]
  Wmin = Wmax/HRSLRS
  return [Wmax,Wmin]

HRSLRS = 3.006
Pl = 0.015
Gmax = 3.006
Gmin = 1
proportion = 0.1
type1 = "unelectroformed"
type2 = "stuck_at_G_min"
type3 = "stuck_at_G_max"
i = 1
tanh_accuracy_cols = ["Undistrubed accuracy","Type1 disturbed","Type2 disturbed","Type3 disturbed"]
tanh_accuracy = []

while i <= 30:
  train_images = mnist.train_images()
  train_labels = mnist.train_labels()
  test_images = mnist.test_images()
  test_labels = mnist.test_labels()

  # Normalize the images.
  train_images = (train_images / 255) - 0.5
  test_images = (test_images / 255) - 0.5

  # Flatten the images.
  train_images = train_images.reshape((-1, 784))
  test_images = test_images.reshape((-1, 784))

  # Build the Tanh model.
  model = Sequential([
    Dense(100, activation='tanh', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])

  # Load the model's saved weights.
  model.load_weights("Tanh_files/Tanh_model_"+str(i)+".h5")

  # Select weights for layers
  weights = model.get_weights()
  # Weights for hidden layer
  layer_1_weights = weights[0]
  # Weights for output layer
  layer_2_weights = weights[2]

  max_weight_layer_1 = max_and_min_weights(layer_1_weights,HRSLRS,Pl)[0]
  max_weight_layer_2 = max_and_min_weights(layer_2_weights,HRSLRS,Pl)[0]

  layer_1_conductances = badmemristor.map.w_to_G(layer_1_weights,max_weight_layer_1,Gmin,Gmax)
  layer_2_conductances = badmemristor.map.w_to_G(layer_2_weights,max_weight_layer_2,Gmin,Gmax)

  disturbed_type1_layer_1_conductances = badmemristor.nonideality.D2D.faulty(layer_1_conductances,proportion,type1)
  disturbed_type2_layer_1_conductances = badmemristor.nonideality.D2D.faulty(layer_1_conductances,proportion,type2)
  disturbed_type3_layer_1_conductances = badmemristor.nonideality.D2D.faulty(layer_1_conductances,proportion,type3)

  disturbed_type1_layer_2_conductances = badmemristor.nonideality.D2D.faulty(layer_2_conductances,proportion,type1)
  disturbed_type2_layer_2_conductances = badmemristor.nonideality.D2D.faulty(layer_2_conductances,proportion,type2)
  disturbed_type3_layer_2_conductances = badmemristor.nonideality.D2D.faulty(layer_2_conductances,proportion,type3)

  disturbed_type1_layer_1_weights = badmemristor.map.G_to_w(disturbed_type1_layer_1_conductances,max_weight_layer_1,Gmax)
  disturbed_type2_layer_1_weights = badmemristor.map.G_to_w(disturbed_type2_layer_1_conductances,max_weight_layer_1,Gmax)
  disturbed_type3_layer_1_weights = badmemristor.map.G_to_w(disturbed_type3_layer_1_conductances,max_weight_layer_1,Gmax)

  disturbed_type1_layer_2_weights = badmemristor.map.G_to_w(disturbed_type1_layer_2_conductances,max_weight_layer_2,Gmax)
  disturbed_type2_layer_2_weights = badmemristor.map.G_to_w(disturbed_type2_layer_2_conductances,max_weight_layer_2,Gmax)
  disturbed_type3_layer_2_weights = badmemristor.map.G_to_w(disturbed_type3_layer_2_conductances,max_weight_layer_2,Gmax)

  tanh_type1_model_weights = [disturbed_type1_layer_1_weights,weights[1],disturbed_type1_layer_2_weights,weights[3]]
  tanh_type2_model_weights = [disturbed_type2_layer_1_weights,weights[1],disturbed_type2_layer_2_weights,weights[3]]
  tanh_type3_model_weights = [disturbed_type3_layer_1_weights,weights[1],disturbed_type3_layer_2_weights,weights[3]]

  tanh_type1_model = Sequential([
    Dense(100, activation='tanh', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])
  tanh_type2_model = Sequential([
    Dense(100, activation='tanh', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])
  tanh_type3_model = Sequential([
    Dense(100, activation='tanh', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])


  tanh_type1_model.set_weights(tanh_type1_model_weights)
  tanh_type2_model.set_weights(tanh_type2_model_weights)
  tanh_type3_model.set_weights(tanh_type3_model_weights)

  model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )
  tanh_type1_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )
  tanh_type2_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )
  tanh_type3_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )

  var = model.evaluate(
    test_images,
    to_categorical(test_labels)
  )
  var1 = tanh_type1_model.evaluate(
    test_images,
    to_categorical(test_labels)
  )
  var2 = tanh_type2_model.evaluate(
    test_images,
    to_categorical(test_labels)
  )
  var3 = tanh_type3_model.evaluate(
    test_images,
    to_categorical(test_labels)
  )
  tanh_type1_filename = "Type1_disturbed/Tanh_type1_disturbed_files/Tanh_type1_disturbed_model_" + str(i) + ".h5"
  tanh_type2_filename = "Type2_disturbed/Tanh_type2_disturbed_files/Tanh_type1_disturbed_model_" + str(i) + ".h5"
  tanh_type3_filename = "Type3_disturbed/Tanh_type3_disturbed_files/Tanh_type1_disturbed_model_" + str(i) + ".h5"

  tanh_type1_model.save_weights(tanh_type1_filename)
  tanh_type2_model.save_weights(tanh_type2_filename)
  tanh_type3_model.save_weights(tanh_type3_filename)

  tanh_ith_accuracy = [var[1],var1[1],var2[1],var3[1]]
  tanh_accuracy.append(tanh_ith_accuracy)
  i += 1

with open("Tanh_accuracy.csv","w",newline="") as f:
  thewriter = csv.writer(f)
  thewriter.writerow(["Undisturbed","Type1 disturbed","Type2 disturbed","Type3 disturbed"])

  for i in range(0,30):
    thewriter.writerow(tanh_accuracy[i])