import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    """
    模型数据保存类，从每个 epoch 中找到各个 batch 中最大的准确度，将其保存，同时还能找到最小的损失。
    同时，在每个 epoch 结束后调用检测函数，并记录损失值和准确率。
    """

    def __init__(self, test_data, test_labels):
        super(CustomCallback, self).__init__()
        self.test_data = test_data
        self.test_labels = test_labels
        self.max_custom_accuracy = []
        self.min_loss = []
        self.epoch_losses = []  # 存储每个 epoch 的损失
        self.epoch_accuracies = []  # 存储每个 epoch 的准确率
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

        # 在每个 epoch 结束时调用检测函数，并记录损失和准确率
        loss, accuracy = self.model.evaluate(
            self.test_data, self.test_labels, verbose=0
        )
        self.epoch_losses.append(loss)
        self.epoch_accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}: 测试集损失: {loss}, 准确率: {accuracy}")
