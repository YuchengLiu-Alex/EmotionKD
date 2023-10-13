from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras.metrics import categorical_accuracy
import tensorflow as tf
from tensorflow_addons.losses import npairs_loss
from tensorflow.keras.layers import Activation


# tf.compat.v1.disable_eager_execution()

class master_Loss(KL.Layer):
    def __init__(self, e1=1, e2=1, **kwargs):
        super(master_Loss, self).__init__(**kwargs)
        self.e1 = e1
        self.e2 = e2
        self.Hard = tf.keras.losses.categorical_crossentropy
        self.student_loss = tf.keras.losses.categorical_crossentropy

    def call(self, inputs, **kwargs):
        """
        # inputs：true_label, S_soft_label, output
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, S_soft_label, output = inputs

        true_loss = self.e1 * self.Hard(true_label, output)
        S_soft_loss = self.e2 * self.student_loss(true_label, S_soft_label)

        true_loss = K.mean(true_loss)
        S_soft_loss = K.mean(S_soft_loss)

        self.add_loss(true_loss, inputs=True)
        self.add_metric(true_loss, aggregation="mean", name="Master_loss")

        self.add_loss(S_soft_loss, inputs=True)
        self.add_metric(S_soft_loss, aggregation="mean", name="Student_loss")

        self.add_metric(categorical_accuracy(true_label, output[0]), name="acc")

        return output


class MLKDLoss(KL.Layer):
    def __init__(self, e1=1, e2=1, e3=1, **kwargs):
        super(MLKDLoss, self).__init__(**kwargs)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.intermediate = tf.keras.losses.MeanSquaredError()
        self.hard = tf.keras.losses.categorical_crossentropy
        self.soft = tf.keras.losses.KLDivergence()

    def call(self, inputs, **kwargs):
        """
        # inputs：true_label, soft_label, middle_teacher, middle_student, output
        如上，父类KL.Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        true_label, soft_label, middle_teacher, middle_student, output = inputs

        true_loss = self.e1 * self.hard(true_label, output)
        soft_loss = self.e2 * self.soft(soft_label, output)
        middle_loss = self.e3 * self.intermediate(middle_teacher, middle_student)

        true_loss = K.mean(true_loss)
        soft_loss = K.mean(soft_loss)

        self.add_loss(true_loss, inputs=True)
        self.add_metric(true_loss, aggregation="mean", name="true_loss")

        self.add_loss(soft_loss, inputs=True)
        self.add_metric(soft_loss, aggregation="mean", name="soft_loss")

        self.add_loss(middle_loss, inputs=True)
        self.add_metric(middle_loss, aggregation="mean", name="middle_loss")

        self.add_metric(categorical_accuracy(true_label, output), name="acc")

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'e1': self.e1,
            'e2': self.e2,
            "e3": self.e3
        })
        return config
