import tensorflow as tf


# 定义 RMSE 损失函数
def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
