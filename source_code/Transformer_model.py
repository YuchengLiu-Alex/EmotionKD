import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        # print(f"attn_output: {attn_output.shape}")
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # print(f"ffn_output: {ffn_output.shape}")
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        self.config = {
            # "att": self.att,
            # "ffn": self.ffn,
            # "layernorm1": self.layernorm1,
            # "layernorm2": self.layernorm2,
            # "dropout1": self.dropout1,
            # "dropout2": self.dropout2
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        }
        return self.config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        # return positions + x

    def get_config(self):
        self.config = {
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb
        }
        return self.config


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    def get_config(self):
        self.config = {
            # "pos_emb": self.pos_emb
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim
        }
        return self.config


class transformer_feature_extractor(layers.Layer):
    def __init__(self, maxlen=64, embed_dim=64, num_heads=8, ff_dim=64):
        super().__init__()

        self.maxlen=maxlen
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.embedding_layer = PositionEmbedding(maxlen=maxlen, embed_dim=embed_dim)
        self.trans_block_0 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.trans_block_1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.trans_block_2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)

    def call(self, x):
        # print("---")
        # print(x.shape)
        x_0 = self.embedding_layer(x)
        # x_0 = x
        # print(x_0.shape)
        # print(x_0.shape)
        # print("---")
        x_1 = self.trans_block_0(x_0)
        # print(f"X_1: {x_1.shape}")
        x_2 = self.trans_block_1(x_1)
        # print(f"x_2: {x_2.shape}")
        x_3 = self.trans_block_2(x_2)
        # print(f"x_3: {x_3.shape}")
        # print(x_1.shape, x_2.shape, x_3.shape)

        return x_1, x_2, x_3

    def get_config(self):
        self.config = {
            # "embedding_layer": self.embedding_layer,
            # "trans_block_0": self.trans_block_0,
            # "trans_block_1": self.trans_block_1,
            # "trans_block_2": self.trans_block_2,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "maxlen": self.maxlen
        }
        return self.config

def transformer_feature_extracting(input_tensor, maxlen=64, embed_dim=64, num_heads=8, ff_dim=64):
    x_0 = PositionEmbedding(maxlen=maxlen, embed_dim=embed_dim)(input_tensor)
    x_1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(x_0)
    x_2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(x_1)
    x_3 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(x_2)
    return x_1, x_2, x_3
