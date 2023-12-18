import torch  # pytorch框架
import pandas as pd  # 数据集读入库
import matplotlib.pyplot as plt  # 绘制图表
import torch.nn as nn  # 模型中的网络层使用
from torch import optim  # 优化器使用
from torch.utils.data import DataLoader, TensorDataset  # 数据集处理
from sklearn.model_selection import train_test_split  # 数据划分
import read as r
import data_process as dp


def main():
    # 输入四个文件的路径
    # file_m=input()
    # file_r=input()
    # file_t=input()
    # file_l=input()
    file_m = "./data/movies.csv"
    file_r = "./data/ratings.csv"
    file_t = "./data/tags.csv"
    file_l = "./data/links.csv"

    # 读入数据
    movie_df, ratings_df, tags_df, links_df = r.read_csv_file(
        file_m, file_r, file_t, file_l
    )

    # 处理数据
    df = dp.solve_data(movie_df, ratings_df, tags_df)
    # df.to_excel("create_data/df_data.xlsx", index=False)


if __name__ == "__main__":
    main()
