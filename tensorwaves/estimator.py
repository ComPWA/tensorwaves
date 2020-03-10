import tensorflow as tf


class UnbinnedLL(tf.keras.losses.Loss):
    def __init__(self):
        super(UnbinnedLL, self).__init__(name='UnbinnedLL')

    def call(self, y_true, y_pred):
        logs = tf.math.log(y_pred)
        ll = tf.reduce_sum(logs)
        return ll
