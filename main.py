import tensorflow as tf

from genetic import Genetic

if __name__ == '__main__':
    train, test = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz")
    x_train, y_train = train
    x_test, y_test = test

    gtc = Genetic(
        input_shape=36,
        output_shape=1,
        data=x_train,
        labels=x_train,
        genome_len_max=3,
        genome_len_min=1,
        genome_width_max=30,
        genome_width_min=5,
    )
    gtc.evolution(10)
