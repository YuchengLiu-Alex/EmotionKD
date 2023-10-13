import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


class Classifier(layers.Layer):
    def __init__(self, hide_dim, output_dim):
        super().__init__()

        self.hide_dim = hide_dim
        self.output_dim = output_dim

        self.dense_0 = layers.Dense(hide_dim)
        self.dense_1 = layers.Dense(hide_dim)
        self.dense_2 = layers.Dense(hide_dim)
        self.final_dense = layers.Dense(output_dim)
        self.dropout_1 = layers.Dropout(0.5)
        self.dropout_2 = layers.Dropout(0.5)

    def call(self, inputs):
        x = self.dropout_1(inputs)
        x = self.dense_0(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        # x = self.dropout_2(x)
        x = self.final_dense(x)
        return x

    def get_config(self):
        self.config = {
            # "dense_0": self.dense_0,
            # "dense_1": self.dense_1,
            # "dense_2": self.dense_2,
            # "final_dense": self.final_dense,
            "hide_dim": self.hide_dim,
            "output_dim": self.output_dim
        }
        return self.config



class EEG_Head(layers.Layer):
    def __init__(self):
        super().__init__()
        self.padding = "same"
        self.activation = tf.nn.relu
        self.temporal_cnn_layer = layers.Conv2D(kernel_size=(1, 32), filters=64, strides=(1, 8),
                                                kernel_regularizer=keras.regularizers.l2(0.001))
        self.activate_layer = layers.Activation(activation=self.activation)
        self.bm = layers.BatchNormalization()
        self.maxpooling = layers.MaxPool2D(pool_size=(1, 8), strides=8)
        self.spatoal_fusion_layer = layers.Conv2D(kernel_size=(28, 1), filters=64, strides=(1, 1),
                                                  kernel_regularizer=keras.regularizers.l2(0.001))
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs):
        x = self.temporal_cnn_layer(inputs)
        x = self.bm(x)
        x = self.activate_layer(x)
        # print(f"CNN_1_output: {x.shape}")
        x = self.spatoal_fusion_layer(x)
        x = tf.squeeze(x, axis=1)
        x = self.dropout(x)
        # print(f"CNN_2_output: {x.shape}")
        return x

    def get_config(self):
        self.config = {
            # "temporal_cnn_layer": self.temporal_cnn_layer,
            # "activate_layer": self.activate_layer,
            # "bm": self.bm,
            # "maxpooling": self.maxpooling,
            # "spatoal_fusion_layer": self.spatoal_fusion_layer,
            # "reshape": self.reshape
        }
        return self.config


class GSR_Head(layers.Layer):
    def __init__(self):
        super().__init__()
        self.padding = "same"
        self.activation = tf.nn.relu
        self.temporal_cnn_layer = layers.Conv2D(kernel_size=(1, 32), filters=64, strides=(1, 8),
                                                kernel_regularizer=keras.regularizers.l2(0.001))
        self.activate_layer = layers.Activation(activation=self.activation)
        self.bm = layers.BatchNormalization()
        self.maxpooling = layers.MaxPool2D(pool_size=(1, 8), strides=8)
        self.spatoal_fusion_layer = layers.Conv2D(kernel_size=(1, 1), filters=64, strides=(1, 1),
                                                  kernel_regularizer=keras.regularizers.l2(0.001))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs):
        x = self.temporal_cnn_layer(inputs)
        x = self.bm(x)
        x = self.activate_layer(x)
        # print(f"CNN_1_output: {x.shape}")
        x = self.spatoal_fusion_layer(x)
        x = tf.squeeze(x, axis=1)
        x = self.dropout(x)
        # print(f"CNN_2_output: {x.shape}")
        return x

    def get_config(self):
        self.config = {
            # "temporal_cnn_layer": self.temporal_cnn_layer,
            # "activate_layer": self.activate_layer,
            # "bm": self.bm,
            # "maxpooling": self.maxpooling,
            # "spatoal_fusion_layer": self.spatoal_fusion_layer
        }
        return self.config

def Classifier_Function(inputs, hide_dim, output_dim):
    dense_0 = layers.Dense(hide_dim, name="classifier_input")
    dense_1 = layers.Dense(hide_dim)
    dense_2 = layers.Dense(hide_dim)
    final_dense = layers.Dense(output_dim, name="logit_layer")
    dropout_1 = layers.Dropout(0.5)

    x = dropout_1(inputs)
    x = dense_0(x)
    x = dense_1(x)
    x = dense_2(x)
    x = final_dense(x)

    return x

