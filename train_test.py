import torch  # pytorch框架
import pandas as pd  # 数据集读入库
import matplotlib.pyplot as plt  # 绘制图表
import torch.nn as nn  # 模型中的网络层使用
from torch import optim  # 优化器使用
from torch.utils.data import DataLoader, TensorDataset  # 数据集处理
from sklearn.model_selection import train_test_split  # 数据划分


def features_target(df):
    """
    划分数据为特征列和目标列，目标列是rating
    :param df: 数据集
    :return: f：特征列，t：目标列
    """
    # 数据划分选取特征列和目标列
    features = df.iloc[:, df.columns != "rating"].values
    target = df.iloc[:, df.columns == "rating"].values
    return features, target


def train_test(df, test_size, random_state):
    """
    划分训练集和测试集
    :param df: 数据集
    :param test_size: 测试集占的比例
    :param random_state: 随机分配数
    :return: 测试集和训练集的自变量因变量
    """
    f, t = features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        f, t, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test
