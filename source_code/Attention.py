from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ScoreGenerationBlock(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense_0 = layers.Dense(input_dim)
        self.dense_1 = layers.Dense(output_dim)
        self.bm_0 = layers.BatchNormalization()
        self.bm_1 = layers.BatchNormalization()
        self.activate_0 = layers.Activation(activation=tf.nn.relu)
        self.activate_1 = layers.Activation(activation=tf.nn.softsign)

    def call(self, inputs):
        x = self.dense_0(inputs)
        x = self.bm_0(x)
        x = self.activate_0(x)
        x = self.dense_1(x)
        x = self.bm_1(x)
        x = self.activate_1(x)
        return x

    def get_config(self):
        self.config = {
            # "dense_0": self.dense_0,
            # "dense_1": self.dense_1,
            # "bm_0": self.bm_0,
            # "bm_1": self.bm_1,
            # "activate_0": self.activate_0,
            # "activate_1": self.activate_1,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }
        return self.config


class MultiModalFusion(layers.Layer):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.concatenate = layers.Concatenate(axis=1)
        self.mm_block_before = ScoreGenerationBlock(input_dim, output_dim)
        self.mm_block_after = ScoreGenerationBlock(input_dim, output_dim)
        self.multiply = layers.Multiply()

    def call(self, inputs):
        EEG_input_before, GSR_input_before, EEG_input_after, GSR_input_after = inputs

        feature_before = self.concatenate([EEG_input_before, GSR_input_before])
        feature_after = self.concatenate([EEG_input_before, GSR_input_after])
        # print(feature_before.shape, feature_after.shape)
        score_before = self.mm_block_before(feature_before)
        score_after = self.mm_block_after(feature_after)
        # print(score_before.shape, score_after.shape)
        att_feature_before = self.multiply([score_before, feature_before])
        att_feature_after = self.multiply([score_after, feature_after])

        return att_feature_after + att_feature_before

    def get_config(self):
        self.config = {
            # "concatenate": self.concatenate,
            # "mm_block_before": self.mm_block_before,
            # "mm_block_after": self.mm_block_after,
            # "multiply": self.multiply,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }
        return self.config


class MultiDepthFusion(layers.Layer):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.concatenate = layers.Concatenate()
        self.sa_block_after = ScoreGenerationBlock(input_dim, output_dim)
        self.sa_block_intermediate = ScoreGenerationBlock(input_dim, output_dim)
        self.sa_block_before = ScoreGenerationBlock(input_dim, output_dim)
        self.multiply = layers.Multiply()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        input_before, input_intermediate, input_after = inputs

        score_before = self.sa_block_before(input_before)
        score_after = self.sa_block_after(input_after)
        score_intermediate = self.sa_block_intermediate(input_intermediate)

        att_feature_before = self.multiply([score_before, input_before])
        att_feature_after = self.multiply([score_after, input_after])
        att_feature_intermediate = self.multiply([score_intermediate, input_intermediate])

        # print(att_feature_intermediate.shape, att_feature_before.shape, att_feature_after.shape)
        # print((att_feature_after + att_feature_before + att_feature_intermediate).shape)

        return att_feature_after + att_feature_before + att_feature_intermediate

    def get_config(self):
        self.config = {
            # "concatenate": self.concatenate,
            # "sa_block_after": self.sa_block_after,
            # "sa_block_intermediate": self.sa_block_intermediate,
            # "sa_block_before": self.sa_block_before,
            # "multiply": self.multiply
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }
        return self.config
