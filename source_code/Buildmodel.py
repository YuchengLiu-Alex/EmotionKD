from tensorflow import keras
from tensorflow.keras import layers, Model

from Attention import MultiDepthFusion, MultiModalFusion
from BasicModule import Classifier, EEG_Head, GSR_Head
from Distill_zoo import MLKDLoss
from Transformer_model import transformer_feature_extractor, PositionEmbedding, TransformerBlock, \
    transformer_feature_extracting


def get_GSR_model(KD_mode="MLKD", learning_rate=1e-6, e2=0.1, e3=1):
    input_ori_gsr = layers.Input((1, 512, 1), name="GSR_input")
    input_hard_label = layers.Input(2, name="hard_label")
    input_soft_label = layers.Input(2, name="soft_label")

    # print(input_ori_gsr.shape)

    gsr_feature = GSR_Head()(input_ori_gsr)

    # print(gsr_feature.shape)

    gsr_feature_1, gsr_feature_2, gsr_feature_3 = transformer_feature_extractor()(gsr_feature)

    # print(gsr_feature_1.shape, gsr_feature_2.shape, gsr_feature_3.shape)

    att_feature = MultiDepthFusion(64, 64)([gsr_feature_1, gsr_feature_2, gsr_feature_3])

    # print(att_feature.shape)

    att_feature = layers.Flatten(name="att_feature")(att_feature)

    # print(att_feature.shape)

    logit_feature = Classifier(hide_dim=128, output_dim=2)(att_feature)

    # print(logit_feature.shape)

    cls = layers.Activation(activation="softmax", name="cls")(logit_feature)

    # print(cls.shape)

    input_soft_feature = layers.Input(att_feature.shape[1], name="soft_feature")

    # model = Model(input_ori_gsr, [cls, logit_feature, att_feature], name="gsr_student")
    # model.summary()

    custom_dir = {
        "GSR_Head": GSR_Head,
        "MultiDepthFusion": MultiDepthFusion,
        "transformer_feature_extractor": transformer_feature_extractor,
        "Classifier": Classifier,
        "PositionEmbedding": PositionEmbedding,
        "TransformerBlock": TransformerBlock
    }

    if KD_mode == "MLKD":
        print("Training with MLKD")

        # For Final Training which is default experiment setting
        # cls = MLKDLoss(e2=0.1, e3=1)([input_hard_label, input_soft_label, input_soft_feature, att_feature, cls])

        # For other Training
        cls = MLKDLoss(e2=e2, e3=e3)([input_hard_label, input_soft_label, input_soft_feature, att_feature, cls])

        model = Model([input_ori_gsr, input_hard_label, input_soft_label, input_soft_feature], cls)
        model_opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=model_opt)
        custom_dir["MLKDLoss"] = MLKDLoss

    model.summary()

    return model, custom_dir


def get_MultiModal_model(learning_rate=5e-5):

    input_ori_gsr = layers.Input((1, 512, 1))
    input_ori_eeg = layers.Input((28, 512, 1))

    # print(input_ori_gsr.shape)

    gsr_feature = GSR_Head()(input_ori_gsr)
    eeg_feature = EEG_Head()(input_ori_eeg)

    # print(gsr_feature.shape, eeg_feature.shape)

    eeg_feature_1, eeg_feature_2, eeg_feature_3 = transformer_feature_extracting(eeg_feature)
    gsr_feature_1, gsr_feature_2, gsr_feature_3 = transformer_feature_extracting(gsr_feature)

    # print(gsr_feature_1.shape, gsr_feature_2.shape, gsr_feature_3.shape)
    # print(eeg_feature_1.shape, eeg_feature_2.shape, eeg_feature_3.shape)

    att_feature_1 = MultiModalFusion(64, 64)([eeg_feature_1, gsr_feature_1, eeg_feature_2, gsr_feature_2])
    att_feature_2 = MultiModalFusion(64, 64)([eeg_feature_2, gsr_feature_2, eeg_feature_3, gsr_feature_3])
    att_feature = att_feature_1 + att_feature_2

    # att_feature = att_feature_1

    # print(att_feature.shape)
    att_feature = layers.Flatten(name="att_feature")(att_feature)
    # print(att_feature.shape)
    logit_feature = Classifier(hide_dim=128, output_dim=2)(att_feature)
    # print(logit_feature.shape)
    cls = layers.Activation(activation="softmax", name="prediction")(logit_feature)
    # print(cls.shape)

    model = Model([input_ori_eeg, input_ori_gsr], [cls, logit_feature, att_feature])
    model.summary()
    model_opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=model_opt, loss=["categorical_crossentropy", None, None], metrics=["acc"])

    custom_dir = {
        "GSR_Head": GSR_Head,
        "MultiModalFusion": MultiModalFusion,
        "transformer_feature_extractor": transformer_feature_extractor,
        "Classifier": Classifier,
        "PositionEmbedding": PositionEmbedding,
        "TransformerBlock": TransformerBlock,
        "EEG_Head": EEG_Head
    }

    # model.summary()

    return model, custom_dir

if __name__ == "__main__":
    get_MultiModal_model()
    get_GSR_model()
