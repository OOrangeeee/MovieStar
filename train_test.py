from sklearn.model_selection import train_test_split
import numpy as np


def features_target(df):
    """
      划分数据为特征列和目标列，特征列包括movie_id, user_id, genres, tag，目标列是rating
    :param df: 数据集
    :return: 特征列和目标列
    """
    # 假设电影ID、用户ID、标签和类型列的名称分别是'movie_id', 'user_id', 'tag'
    movie_id = df["movieId"].values
    user_id = df["userId"].values
    tag = df["tag"].values

    # 电影类型的20列合并为一个数组
    not_genres_columns = ["movieId", "title", "userId", "rating", "tag"]  # 非特征列
    all_columns = df.columns.tolist()  # 全部列
    genres_columns = [
        col for col in all_columns if col not in not_genres_columns
    ]  # 特征列表格
    genres = df[genres_columns].values

    rating = df["rating"].values  # 目标列
    return [movie_id, user_id, tag, genres], rating


def train_test(df, test_size, random_state):
    """
    划分训练集和测试集
    :param df: 数据集
    :param test_size: 测试集占的比例
    :param random_state: 随机分配数
    :return: 训练集和测试集的数据
    """
    df = normalize_ratings(df, "rating")
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    x_train, y_train = features_target(df_train)
    x_test, y_test = features_target(df_test)

    # for i in range(len(x_train)):
    #     filename = "create_data/x_train_" + str(i) + ".csv"
    #     np.savetxt(filename, x_train[i], delimiter=",")
    # np.savetxt("create_data/y_train.csv", y_train, delimiter=",")
    # for i in range(len(x_test)):
    #     filename = "create_data/x_test_" + str(i) + ".csv"
    #     np.savetxt(filename, x_test[i], delimiter=",")
    # np.savetxt("create_data/y_test.csv", y_test, delimiter=",")

    return x_train, x_test, y_train, y_test


def normalize_ratings(df, rating_column="rating"):
    """
    对评分进行最小-最大归一化处理。

    :param df: 包含评分的DataFrame。
    :param rating_column: 存储评分的列名。
    :return: 归一化后的DataFrame。
    """
    min_rating = df[rating_column].min()
    max_rating = df[rating_column].max()
    df[rating_column] = (df[rating_column] - min_rating) / (max_rating - min_rating)
    return df
