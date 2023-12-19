import model as m
import train_test as tt
from call_back import CustomCallback
from draw import draw_show_save


def solve_ncf(
    df,
    batch_size=32,
    embedding_size=50,
    test_size=0.1,
    learn_rate=0.001,
):
    """
    针对NCF模型的解决方案
    :param df: 数据集
    :param batch_size: 单次训练量
    :param embedding_size: 嵌入维度
    :param test_size: 测试集占比
    :param learn_rate: 学习率
    :return: 无
    """
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
        learn_rate,
    )
    epochs = 5
    train(df, test_size, 108, batch_size, ncf, epochs, 1)
    train(df, test_size, 568, batch_size, ncf, epochs, 2)
    train(df, test_size, 782, batch_size, ncf, epochs, 3)


def train(df, test_size, random_state, batch_size, model_ncf, epochs, num=1):
    # 分配数据集
    x_train, x_test, y_train, y_test = tt.train_test(df, test_size, random_state)

    # 更改数据形状
    for i in range(3):
        x_train[i] = x_train[i].reshape(-1, 1)
        x_test[i] = x_test[i].reshape(-1, 1)
    # 构建回溯示例
    callback = CustomCallback(x_test, y_test)

    # 训练模型
    model_ncf.train(x_train, y_train, callback, epochs, batch_size)

    # 输出
    print("训练集" + str(num) + "的最大准确率：", callback.max_custom_accuracy)
    print(
        "训练集" + str(num) + "的平均准确率：",
        sum(callback.max_custom_accuracy) / len(callback.max_custom_accuracy),
    )

    print("训练集" + str(num) + "的最小损失：", callback.min_loss)
    print(
        "训练集" + str(num) + "的平均损失：",
        sum(callback.min_loss) / len(callback.min_loss),
    )

    print("测试集" + str(num) + "的准确率", callback.epoch_accuracies)
    print(
        "测试集" + str(num) + "的平均准确率：",
        sum(callback.epoch_accuracies) / len(callback.epoch_accuracies),
    )

    print("测试集" + str(num) + "的损失", callback.epoch_losses)
    print(
        "测试集" + str(num) + "的平均损失：",
        sum(callback.epoch_losses) / len(callback.epoch_losses),
    )

    with open("./create_data/result/res/res_" + str(num) + ".txt", "w") as file:
        file_max_custom_accuracy = [str(i) for i in callback.max_custom_accuracy]
        file_min_loss = [str(i) for i in callback.min_loss]
        file_epoch_accuracies = [str(i) for i in callback.epoch_accuracies]
        file_epoch_losses = [str(i) for i in callback.epoch_losses]

        file.write("训练集" + str(num) + "的最大准确值：")
        file.write(",".join(file_max_custom_accuracy))
        file.write("\n")
        file.write("训练集准确率" + str(num) + "的平均值：")
        file.write(
            str(sum(callback.max_custom_accuracy) / len(callback.max_custom_accuracy))
        )
        file.write("\n")

        file.write("\n")

        file.write("训练集" + str(num) + "的最小损失：")
        file.write(" , ".join(file_min_loss))
        file.write("\n")
        file.write("训练集损失" + str(num) + "的平均值：")
        file.write(str(sum(callback.min_loss) / len(callback.min_loss)))
        file.write("\n")

        file.write("\n")

        file.write("测试集" + str(num) + "的准确率：")
        file.write(" , ".join(file_epoch_accuracies))
        file.write("\n")
        file.write("训练集损失" + str(num) + "的平均值：")
        file.write(str(sum(callback.epoch_accuracies) / len(callback.epoch_accuracies)))
        file.write("\n")

        file.write("\n")

        file.write("测试集" + str(num) + "的损失：")
        file.write(" , ".join(file_epoch_losses))
        file.write("\n")
        file.write("训练集损失" + str(num) + "的平均值：")
        file.write(str(sum(callback.epoch_losses) / len(callback.epoch_losses)))
        file.write("\n")

    # 绘图
    draw_show_save(
        epochs,
        callback.max_custom_accuracy,
        "训练次数",
        "最大准确度",
        "第 " + str(num) + " 次训练中准确度变化曲线",
    )
    draw_show_save(
        epochs, callback.min_loss, "训练次数", "最小损失", "第 " + str(num) + " 训练中损失变化曲线"
    )
    draw_show_save(
        epochs,
        callback.epoch_accuracies,
        "测试次数",
        "测试准确度",
        "第 " + str(num) + " 测试中准确度变化曲线",
    )
    draw_show_save(
        epochs, callback.epoch_losses, "测试次数", "测试损失值", "第 " + str(num) + " 测试中损失变化曲线"
    )
