# Loading required libraries
library(tensorflow)
library(keras)
library(reticulate)
library(ggplot2)
library(dplyr)

# Loading the MNIST dataset and spliting it into training and testing data and labels
mnist <- dataset_mnist()
trainx <- mnist$train$x
trainy <- mnist$train$y
testx <- mnist$test$x
testy <- mnist$test$y

# Converting 28*28 images into 1D 784 vectors and normalize
trainx <- array_reshape(trainx, c(nrow(trainx), 784)) / 255
testx <- array_reshape(testx, c(nrow(testx), 784)) / 255

trainy <- to_categorical(trainy, 10)
testy <- to_categorical(testy, 10)

# Visualizing the data
par(mfrow=c(3,3))
for(i in 1:9) {
  img <- matrix(trainx[i, ], nrow = 28, ncol = 28)
  plot(as.raster(img, max = 1))
}

# Define custom layer for the forward forward algorithm                   
FFlayerdense <- setRefClass("FFlayerdense", contains = "Layer",
                            methods = list(
                              initialize = function(units, optimizer, loss_metric, num_epochs = 50,
                                                    use_bias = TRUE, kernel_initializer = "glorot_uniform",
                                                    bias_initializer = "zeros", kernel_regularizer = NULL,
                                                    bias_regularizer = NULL, ...) {
                                callSuper(...)
                                self$dense <- layer_dense(
                                  units = units,
                                  use_bias = use_bias,
                                  kernel_initializer = kernel_initializer,
                                  bias_initializer = bias_initializer,
                                  kernel_regularizer = kernel_regularizer,
                                  bias_regularizer = bias_regularizer
                                )
                                self$relu <- layer_activation("relu")
                                self$optimizer <- optimizer
                                self$loss_metric <- loss_metric
                                self$threshold <- 1.5
                                self$num_epochs <- num_epochs
                              },
                              
                              call = function(x) {
                                x_norm <- tf$norm(x, ord = 2, axis = 1, keepdims = TRUE)
                                x_norm <- x_norm + 1e-4
                                x_dir <- x / x_norm
                                res <- self$dense(x_dir)
                                return(self$relu(res))
                              },
                              
                              forward_forward = function(posx, negx) {
                                for (i in 1:self$num_epochs) {
                                  with(tf$GradientTape(persistent=TRUE) %as% tape, {
                                    posg <- tf$reduce_mean(tf$pow(self$call(posx), 2), axis=1)
                                    negg <- tf$reduce_mean(tf$pow(self$call(negx), 2), axis=1)
                                    loss <- tf$log(1 + tf$exp(tf$concat(list(-posg + self$threshold, negg - self$threshold), axis=0)))
                                    mean_loss <- tf$reduce_mean(loss)
                                  })
                                  gradients <- tape$gradient(mean_loss, self$dense$trainable_weights)
                                  self$optimizer$apply_gradients(zip2(gradients, self$dense$trainable_weights))
                                  self$loss_metric$update_state(mean_loss)
                                }
                                return(list(
                                  tf$stop_gradient(self$call(posx)),
                                  tf$stop_gradient(self$call(negx)),
                                  self$loss_metric$result()
                                ))
                              }
                            )
)  

# Define the FFNetwork model
FFNetwork <- setRefClass("FFNetwork", contains = "Model",
                         fields = list(
                           layer_optimizer = "Optimizer",
                           loss_var = "Tensor",
                           loss_count = "Tensor",
                           layer_list = "list"
                         ),
                         methods = list(
                           initialize = function(dims, layer_optimizer = optimizer_adam(learning_rate = 0.03), ...) {
                             callSuper(...)
                             self$layer_optimizer <- layer_optimizer
                             self$loss_var <- tf$Variable(0.0, trainable = FALSE, dtype = tf$float32)
                             self$loss_count <- tf$Variable(0.0, trainable = FALSE, dtype = tf$float32)
                             self$layer_list <- list()
                             for (d in dims) {
                               self$layer_list <- c(self$layer_list, FFlayerdense(units = d, optimizer = self$layer_optimizer, loss_metric = metric_mean()))
                             }
                           },   
                           
                           overlay_y_on_x = function(data) {
                             X_sample <- data[[1]]
                             y_sample <- data[[2]]
                             max_sample <- tf$reduce_max(X_sample, axis=1, keepdims=TRUE)
                             max_sample <- tf$cast(max_sample, dtype=tf$float64)
                             X_zeros <- tf$zeros(c(10), dtype=tf$float64)
                             X_update <- tf$dynamic_stitch(list(tf$range(10), tf$expand_dims(y_sample, 0)), list(X_zeros, max_sample))
                             X_sample <- tf$concat(list(X_sample, X_update), axis=1)
                             return(list(X_sample, y_sample))
                           },
                           
                           predict_one_sample = function(x) {
                             goodness_per_label <- list()
                             for (label in 0:9) {
                               h <- self$overlay_y_on_x(list(x, label))
                               goodness <- list()
                               for (layer in self$layer_list) {
                                 h <- layer$call(h[[1]])
                                 goodness <- c(goodness, tf$reduce_mean(tf$pow(h, 2), axis=1))
                               }
                               goodness_per_label[[label + 1]] <- tf$expand_dims(tf$reduce_sum(goodness), 1)
                             }
                             goodness_per_label <- tf$concat(goodness_per_label, axis=1)
                             return(tf$cast(tf$argmax(goodness_per_label, 1), tf$int64))
                           },
                           
                           predict = function(data) {
                             preds <- tf$map_fn(self$predict_one_sample, data, dtype=tf$int64)
                             return(tf$convert_to_tensor(preds))
                           },
                           
                           train_step = function(data) {
                             x <- data[[1]]
                             y <- data[[2]]
                             
                             xy_pos <- tf$map_fn(self$overlay_y_on_x, list(x, y), dtype=tf$float64)
                             x_pos <- xy_pos[[1]]
                             y_pos <- xy_pos[[2]]
                             
                             random_y <- tf$random$shuffle(y)
                             xy_neg <- tf$map_fn(self$overlay_y_on_x, list(x, random_y), dtype=tf$float64)
                             x_neg <- xy_neg[[1]]
                             
                             h_pos <- x_pos
                             h_neg <- x_neg
                             
                             for (layer in self$layer_list) {
                               result <- layer$forward_forward(h_pos, h_neg)
                               h_pos <- result[[1]]
                               h_neg <- result[[2]]
                               loss <- result[[3]]
                               self$loss_var$assign_add(loss)
                               self$loss_count$assign_add(1.0)
                             }
                             
                             mean_loss <- self$loss_var / self$loss_count
                             return(mean_loss)
                           }
                         )
)

# Initialize and compile the model
model <- FFNetwork$new(dims = c(784, 500, 500))
model$compile(optimizer = optimizer_adam(learning_rate = 0.03), loss = "mse", metrics = list("accuracy"))

# Train the model
epochs <- 250
history <- model$fit(x = trainx, y = trainy, epochs = epochs, batch_size = 32)

# Evaluate the model
results <- model$evaluate(testx, testy)
cat("Test loss:", results$loss, "\n")
cat("Test accuracy:", results$accuracy, "\n")

# Plot the training history
plot(history)

