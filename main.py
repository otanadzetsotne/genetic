import tensorflow as tf
import numpy as np

from src.genetic_encoder import GeneticEncoder

if __name__ == '__main__':
    train, test = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz")
    x_train, y_train = train
    x_test, y_test = test

    norm = np.linalg.norm(x_train)
    x_train = x_train / norm

    gtc = GeneticEncoder(
        input_shape=13,
        output_shape=3,
        data=x_train,
        labels=x_train,
        genome_len_max=3,
        genome_len_min=1,
        genome_width_max=12,
        genome_width_min=5,
    )
    gtc.evolution(10)
