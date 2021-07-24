import tensorflow as tf

from genetic import Genetic

if __name__ == '__main__':
    train, test = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz")
    x_train, y_train = train
    x_test, y_test = test

    gtc = Genetic(
        input_shape=13,
        output_shape=1,
        data=x_train,
        labels=y_train,
    )
    gtc.evolution(5)
