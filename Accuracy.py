from tensorflow.keras.metrics import Metric
import tensorflow as tf


class CustomAccuracy(Metric):
    """
    准确度修正类，因为NCF得到的时0-1的浮点数，这里将他还原成0-5的浮点数，又因为是一个随机的浮点数不一定是0.5的倍数，所以只要跟目标值相差不大于0.49就算成功
    """

    # 注：这里主要是针对原有的准确度检验函数的重写，故不做过多解释
    def __init__(self, name="custom_accuracy", **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.correct_predictions = self.add_weight(name="cp", initializer="zeros")
        self.total_predictions = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        # 将预测值映射回原始评分范围
        y_pred_rescaled = y_pred * 4.5 + 0.5
        y_true_rescaled = y_true * 4.5 + 0.5
        diff = tf.abs(y_true_rescaled - y_pred_rescaled)
        correct = tf.cast(tf.less_equal(diff, 0.49), "float32")  # 相差不大于0.49就算成功
        self.correct_predictions.assign_add(tf.reduce_sum(correct))
        self.total_predictions.assign_add(tf.cast(tf.size(correct), "float32"))

    def result(self):
        # 计算准确度的百分比
        return (self.correct_predictions / self.total_predictions) * 100

    def reset_state(self):
        self.correct_predictions.assign(0.0)
        self.total_predictions.assign(0.0)
