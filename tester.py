import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# plt.axis([0, 10, 0, 1])

# for i in range(10):
#     y = np.random.random()
#     plt.scatter(i, y)
#     plt.pause(0.20)

# plt.show()

a = tf.constant([1, 2])
print(a)

a = list(a.numpy())
print(a)