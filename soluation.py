import model as m
import train_test as tt
from call_back import CustomCallback


def solve_ncf(
    df, epochs=10, batch_size=32, embedding_size=50, test_size=0.1, random_state=99
):
    """
    针对NCF模型的解决方案
    :param df: 数据集
    :param epochs: 训练次数
    :param batch_size: 单次训练量
    :param embedding_size: 嵌入维度
    :param test_size: 测试集占比
    :param random_state: 测试集划分随机数
    :return: 无
    """

    # 分配数据集
    x_train, x_test, y_train, y_test = tt.train_test(df, test_size, random_state)

    # 计算特征列数量
    not_genres_columns = ["movieId", "title", "userId", "rating", "tag"]  # 非特征列
    all_columns = df.columns.tolist()  # 全部列
    genres_columns = [
        col for col in all_columns if col not in not_genres_columns
    ]  # 特征列表格
    genres_columns_count = len(genres_columns)  # 特征列数

    # 构建模型
    ncf = m.NCFModel(
        df["movieId"].max() + 1,
        df["userId"].max() + 1,
        df["tag"].max() + 1,
        genres_columns_count,
        embedding_size,
    )

    # 构建回溯示例
    callback = CustomCallback()

    # 更改数据形状
    for i in range(3):
        x_train[i] = x_train[i].reshape(-1, 1)
        x_test[i] = x_test[i].reshape(-1, 1)

    # 训练模型
    ncf.train(x_train, y_train, callback, epochs, batch_size)

    # 检验模型
    test_loss, test_accuracy = ncf.evaluate(x_test, y_test)

    # 输出训练时的最大准确度和最小损失函数
    print("Max Accuracy per Epoch:", callback.max_custom_accuracy)
    print("Min Loss per Epoch:", callback.min_loss)

    # 输出测试结果
    print(test_loss)
    print(test_accuracy)
