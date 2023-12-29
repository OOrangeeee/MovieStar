# 最后编辑：
# 晋晨曦 2023.12.15 15:36
# qq：2950171570
# email：Jin0714@outlook.com  回复随缘

import pandas as pd  # 数据集读入库


def read_csv_file(file_movies, file_ratings, file_tags, file_links):
    """
    读取所有的文件到df中
    :param file_movies:电影信息文件
    :param file_ratings:电影评分文件
    :param file_tags:电影标签文件
    :param file_links:电影链接文件
    :return:返回四个DataFrame文件
    """
    movies_df = pd.read_csv(file_movies)
    ratings_df = pd.read_csv(file_ratings)
    tags_df = pd.read_csv(file_tags)
    links_df = pd.read_csv(file_links)
    return movies_df, ratings_df, tags_df, links_df
