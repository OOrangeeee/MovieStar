import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    """
    模型数据保存类，从每个epoch中找到各个batch中最大的准确度，将其保存，同时还能找到最小的损失
    """

    def __init__(self):
        super(CustomCallback, self).__init__()
        self.max_custom_accuracy = []
        self.min_loss = []
        self.current_max_accuracy = 0  # 记录当前 epoch 的最高准确率

    def on_batch_end(self, batch, logs=None):
        current_custom_accuracy = logs.get("custom_accuracy", 0)
        self.current_max_accuracy = max(
            self.current_max_accuracy, current_custom_accuracy
        )

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss", float("inf"))
        # 使用当前 epoch 的最高准确率
        self.max_custom_accuracy.append(self.current_max_accuracy)
        self.min_loss.append(
            min(self.min_loss[-1], current_loss) if self.min_loss else current_loss
        )
        # 重置当前 epoch 的最高准确率
        self.current_max_accuracy = 0
