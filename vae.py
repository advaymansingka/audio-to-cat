from base64 import encode
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
from math import ceil


class VAE(tf.keras.Model):

    def __init__(self, input_size):
        super(VAE, self).__init__()

        self.encoded_size = 10

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        # Replace "pass" statement with your code

        self.encoder = Sequential([
            Flatten(),
            Dense(input_size, activation = 'relu'),
            Dense(ceil(input_size / 2), activation = 'relu'),
            Dense(ceil(input_size / 10), activation = 'relu'),
            Dense(ceil(input_size / 40), activation = 'relu'),
            Dense(self.encoded_size, activation = 'relu'),
        ])


        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, D).             #
        ############################################################################################
        # Replace "pass" statement with your code
        
        self.decoder = Sequential([
            Dense(self.encoded_size, activation = 'relu'),
            Dense(ceil(input_size / 40), activation = 'relu'),
            Dense(ceil(input_size / 10), activation = 'relu'),
            Dense(ceil(input_size / 2), activation = 'relu'),
            Dense(input_size, activation = 'relu'),
        ])

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################


    def call(self, x):
        """
        Performs forward pass through AE model by passing image through 
        encoder, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N,D)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,D)
        """
        x_hat = None

        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code

        # print("call shape:", x.shape)

        encoded_input = self.encoder(x)

        # print("encoded shape:", encoded_input.shape)

        decoded_input = self.decoder(encoded_input)

        # print("decoded input shape", decoded_input.shape)

        x_hat = tf.reshape(decoded_input, x.shape)

        # print("x_hat shape", x_hat.shape)

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat



def mse_loss_function(x_hat, x):

    mse_fn = tf.keras.losses.MeanSquaredError()
    reconstruction_loss = mse_fn(x, x_hat)

    return reconstruction_loss


def loss_function(x_hat, x):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
    Returned loss is the average loss per sample in the current batch.

    Inputs:
    - x_hat: Reconstructed input data of shape (N, D)
    - x: Input data for this timestep of shape (N, D)
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    ################################################################################################
    # TODO: Compute negative variational lowerbound loss as described in the notebook              #
    ################################################################################################
    # Replace "pass" statement with your code

    reconstruction_loss = mse_loss_function(x_hat, x)
    
    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return reconstruction_loss
