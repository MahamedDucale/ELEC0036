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

#Linear param
HRSLRS = 3.006
Pl = 0.015
Gmax = 3.006
Gmin = 1
proportion = 0.1
#Lognormal param
lognormal_G_temp1 = [25., 50., 75., 100., 125.,150.,175.,200.]
lognormal_G_temp2 = [i * (10**3) for i in lognormal_G_temp1]
lognormal_G = np.reciprocal(lognormal_G_temp2)

lognormal_rate = [.40625, .4375, .46875, .59375, .625, .65625, .6875, .71875]

lognormal_mean_temp1 = [1.25, 2., 3., 4., 4., 5., 8., 12.5]
lognormal_mean_temp2 = [(y/100) for y in lognormal_mean_temp1]
lognormal_mean = [np.log(y) for y in lognormal_mean_temp2]

lognormal_sigma_temp1 = [0.4, 0.5, 1., 1.5, 1.5, 2, 2.5, 4.]
lognormal_sigma_temp2 = [(y/100) for y in lognormal_mean_temp1]
lognormal_sigma_temp3 = [np.log(y) for y in lognormal_mean_temp2]
lognormal_sigma = np.ndarray.tolist(np.array(lognormal_mean) - np.array(lognormal_sigma_temp3)) 

type1 = "unelectroformed"
type2 = "stuck_at_G_min"
type3 = "stuck_at_G_max"
type4 = "random_telegraph_noise"
i = 1
sigmoid_accuracy_cols = ["Undistrubed accuracy","Type1 disturbed","Type2 disturbed","Type3 disturbed","Type4 disturbed"]
sigmoid_accuracy = []

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

  # Build the Sigmoid model.
  model = Sequential([
    Dense(100, activation='sigmoid', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])

  # Load the model's saved weights.
  model.load_weights("Sigmoid_files/Sigmoid_model_"+str(i)+".h5")

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
  disturbed_type4_layer_1_conductances = badmemristor.nonideality.model.lognormal(layer_1_conductances, lognormal_G, lognormal_mean, lognormal_sigma, lognormal_rate)

  disturbed_type1_layer_2_conductances = badmemristor.nonideality.D2D.faulty(layer_2_conductances,proportion,type1)
  disturbed_type2_layer_2_conductances = badmemristor.nonideality.D2D.faulty(layer_2_conductances,proportion,type2)
  disturbed_type3_layer_2_conductances = badmemristor.nonideality.D2D.faulty(layer_2_conductances,proportion,type3)
  disturbed_type4_layer_2_conductances = badmemristor.nonideality.model.lognormal(layer_2_conductances, lognormal_G, lognormal_mean, lognormal_sigma, lognormal_rate)

  disturbed_type1_layer_1_weights = badmemristor.map.G_to_w(disturbed_type1_layer_1_conductances,max_weight_layer_1,Gmax)
  disturbed_type2_layer_1_weights = badmemristor.map.G_to_w(disturbed_type2_layer_1_conductances,max_weight_layer_1,Gmax)
  disturbed_type3_layer_1_weights = badmemristor.map.G_to_w(disturbed_type3_layer_1_conductances,max_weight_layer_1,Gmax)
  disturbed_type4_layer_1_weights = badmemristor.map.G_to_w(disturbed_type4_layer_1_conductances,max_weight_layer_1,Gmax)

  disturbed_type1_layer_2_weights = badmemristor.map.G_to_w(disturbed_type1_layer_2_conductances,max_weight_layer_2,Gmax)
  disturbed_type2_layer_2_weights = badmemristor.map.G_to_w(disturbed_type2_layer_2_conductances,max_weight_layer_2,Gmax)
  disturbed_type3_layer_2_weights = badmemristor.map.G_to_w(disturbed_type3_layer_2_conductances,max_weight_layer_2,Gmax)
  disturbed_type4_layer_2_weights = badmemristor.map.G_to_w(disturbed_type4_layer_2_conductances,max_weight_layer_2,Gmax)

  sigmoid_type1_model_weights = [disturbed_type1_layer_1_weights,weights[1],disturbed_type1_layer_2_weights,weights[3]]
  sigmoid_type2_model_weights = [disturbed_type2_layer_1_weights,weights[1],disturbed_type2_layer_2_weights,weights[3]]
  sigmoid_type3_model_weights = [disturbed_type3_layer_1_weights,weights[1],disturbed_type3_layer_2_weights,weights[3]]
  sigmoid_type4_model_weights = [disturbed_type4_layer_1_weights,weights[1],disturbed_type4_layer_2_weights,weights[3]]

  sigmoid_type1_model = Sequential([
    Dense(100, activation='sigmoid', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])
  sigmoid_type2_model = Sequential([
    Dense(100, activation='sigmoid', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])
  sigmoid_type3_model = Sequential([
    Dense(100, activation='sigmoid', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])
  sigmoid_type4_model = Sequential([
    Dense(100, activation='sigmoid', input_shape=(784,)),
    Dense(10, activation='softmax'),
  ])


  sigmoid_type1_model.set_weights(sigmoid_type1_model_weights)
  sigmoid_type2_model.set_weights(sigmoid_type2_model_weights)
  sigmoid_type3_model.set_weights(sigmoid_type3_model_weights)
  sigmoid_type4_model.set_weights(sigmoid_type4_model_weights)

  model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )
  sigmoid_type1_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )
  sigmoid_type2_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )
  sigmoid_type3_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )
  sigmoid_type4_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
  )

  var = model.evaluate(
    test_images,
    to_categorical(test_labels)
  )
  var1 = sigmoid_type1_model.evaluate(
    test_images,
    to_categorical(test_labels)
  )
  var2 = sigmoid_type2_model.evaluate(
    test_images,
    to_categorical(test_labels)
  )
  var3 = sigmoid_type3_model.evaluate(
    test_images,
    to_categorical(test_labels)
  )
  var4 = sigmoid_type4_model.evaluate(
    test_images,
    to_categorical(test_labels)
  )
  sigmoid_type1_filename = "Type1_disturbed/Sigmoid_type1_disturbed_files/Sigmoid_type1_disturbed_model_" + str(i) + ".h5"
  sigmoid_type2_filename = "Type2_disturbed/Sigmoid_type2_disturbed_files/Sigmoid_type2_disturbed_model_" + str(i) + ".h5"
  sigmoid_type3_filename = "Type3_disturbed/Sigmoid_type3_disturbed_files/Sigmoid_type3_disturbed_model_" + str(i) + ".h5"
  sigmoid_type4_filename = "Type4_disturbed/Sigmoid_type4_disturbed_files/Sigmoid_type4_disturbed_model_" + str(i) + ".h5"

  sigmoid_type1_model.save_weights(sigmoid_type1_filename)
  sigmoid_type2_model.save_weights(sigmoid_type2_filename)
  sigmoid_type3_model.save_weights(sigmoid_type3_filename)
  sigmoid_type4_model.save_weights(sigmoid_type4_filename)

  sigmoid_ith_accuracy = [var[1],var1[1],var2[1],var3[1],var4[1]]
  sigmoid_accuracy.append(sigmoid_ith_accuracy)
  i += 1

with open("Sigmoid_accuracy.csv","w",newline="") as f:
  thewriter = csv.writer(f)
  thewriter.writerow(["Undisturbed","Type1 disturbed","Type2 disturbed","Type3 disturbed","Type 4 disturbed"])

  for i in range(0,30):
    thewriter.writerow(sigmoid_accuracy[i])



