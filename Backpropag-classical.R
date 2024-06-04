# 2021A1PS2602P
# SHUBH NEMA

# This code implements artificial neural network using the classical backpropagation approach on the MNIST and CIFAR-10 datasets 

# Load the required libraries
library(keras)
library(reticulate)
use_condaenv("r-reticulate", required = TRUE)

# Load the MNIST dataset
mnist <- dataset_mnist()
traindata <- mnist$train$x
trainlabels <- mnist$train$y
testdata <- mnist$test$x
testlabels <- mnist$test$y

# Convert 28*28 images into 1D 784 vectors and normalize
traindata <- array_reshape(traindata, c(nrow(traindata), 784)) / 255
testdata <- array_reshape(testdata, c(nrow(testdata), 784)) / 255

trainlabels <- to_categorical(trainlabels, 10)
testlabels <- to_categorical(testlabels, 10)

# Create the model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = 'accuracy')

# Fit the model using the training data
history <- model %>% 
  fit(traindata,
      trainlabels,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.2)

# Plot training history
plot(history)

# Evaluate the model on test data
model %>% evaluate(testdata, testlabels)

# Predictions on test data
pred <- model %>% predict(testdata) %>% k_argmax()
pred[1:20]

# NOW USING THE CIFAR-10 DATASET
cifar <- dataset_cifar10()
trainx <- cifar$train$x / 255
testx <- cifar$test$x / 255
trainy <- to_categorical(cifar$train$y, num_classes = 10)
testy <- to_categorical(cifar$test$y, num_classes = 10)

label_name <- c("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# Create the model
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), padding = "same", input_shape = c(32, 32, 3)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512) %>%
  layer_activation("relu") %>%
  layer_dense(units = 10) %>%
  layer_activation("softmax")

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 0.003),
  metrics = "accuracy"
)

# Print the model summary
summary(model)

# Train the model
history <- model %>% fit(
  trainx,
  trainy,
  epochs = 60,
  batch_size = 32,
  validation_split = 0.2
)

# Plot training history
plot(history)

# Evaluate the model on test data
model %>% evaluate(testx, testy)

