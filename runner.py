import argparse
import math
from operator import mod
from pickletools import optimize
from turtle import shape
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.math import sigmoid
from tqdm import tqdm
from vae import VAE, reparametrize, loss_function
import csv


def train_vae(model, data):
    """
    Train your VAE with one epoch.

    Inputs:
    - model: Your VAE instance.
    - train_loader: A tf.data.Dataset of MNIST dataset.
    - args: All arguments.
    - is_cvae: A boolean flag for Conditional-VAE. If your model is a Conditional-VAE,
    set is_cvae=True. If it's a Vanilla-VAE, set is_cvae=False.

    Returns:
    - total_loss: Sum of loss values of all batches.
    """
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    total_loss = 0

    batch_size = 4
    num_batches = data.shape[0] // batch_size

    for i in range(num_batches):

        x_train = data[i*batch_size : (i+1)*batch_size]

        with tf.GradientTape() as tape:

            x_hat, mu, logvar = model.call(x_train)
            loss = loss_function(x_hat, x_train, mu, logvar)

            train_vars = model.trainable_variables
            gradients = tape.gradient(loss, train_vars)
            optimizer.apply_gradients(zip(gradients, train_vars))

        total_loss += loss

    return total_loss


def main():
    # Load MNIST dataset
    with open('testcsv.csv', 'r') as f:
        data = list(csv.reader(f, delimiter=","))
 
    data = np.array(data)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    data = tf.reshape(data, shape=(data.shape[0], 1, data.shape[1], 1))

    # Get an instance of VAE
    model = VAE(data.shape[2])

    # Train VAE
    for epoch_id in range(100):
        total_loss = train_vae(model, data)
        print(f"Train Epoch: {epoch_id} \tLoss: {total_loss/len(data):.6f}")


if __name__ == "__main__":
    main()