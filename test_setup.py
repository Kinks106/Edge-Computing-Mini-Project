import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

# Simple test
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
print("TensorFlow test:", tf.add(a, b))