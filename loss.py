# 最后编辑：
# 晋晨曦 2023.12.19 15:36
# qq：2950171570
# email：Jin0714@outlook.com  回复随缘

import tensorflow as tf


# 定义 RMSE 损失函数
def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
