# 最后编辑：
# 晋晨曦 2023.12.17 9:45
# qq：2950171570
# email：Jin0714@outlook.com  回复随缘

import pandas as pd  # 数据集读入库


def solve_data(movies_df, ratings_df, tags_df):
    """
    把三个数据集合并为一个数据集，并删除时间戳，为每个tag编号，没有tag编号为0，并且将特征列拆开
    :param movies_df:电影信息
    :param ratings_df:打分信息
    :param tags_df:标签信息
    :return:返回结果
    """
    ratings_df = ratings_df.drop(columns=["timestamp"])  # 删除时间戳
    tags_df = tags_df.drop(columns=["timestamp"])  # 删除时间戳
    movies_ratings_df = link_df(
        movies_df, ratings_df, "movieId", "inner"
    )  # 合并电影信息和电影评分
    movies_ratings_tags_df = link_df(
        movies_ratings_df, tags_df, ["movieId", "userId"], "outer"
    )  # 合并评分和标签，这里用outer确保所有的电影标签都载入，但是会出现冗余行
    movies_ratings_tags_df = movies_ratings_tags_df.dropna(
        subset=["rating", "title", "userId", "genres", "movieId"]
    )  # 删除没有信息的冗余行
    movies_ratings_tags_df["tag"] = movies_ratings_tags_df["tag"].fillna(
        "noTags"
    )  # 为没有标签的行填充值

    # 把所有的标签都数字化，没有标签改为0，确保无影响
    encoding = {"noTags": 0}
    unique_values = set(movies_ratings_tags_df["tag"]) - set(encoding.keys())
    for i, value in enumerate(unique_values, start=1):
        encoding[value] = i
    movies_ratings_tags_df["tag"] = movies_ratings_tags_df["tag"].map(encoding)

    # 把特征列分割出来
    genres_split = movies_ratings_tags_df["genres"].str.get_dummies(
        "|"
    )  # 使用get_dummies这个方法将genres这一列的数据分割出来形成新的特征列
    movies_ratings_tags_df = pd.concat(
        [movies_ratings_tags_df, genres_split], axis=1
    )  # 将新生成的列与原始数据合并
    movies_ratings_tags_df = movies_ratings_tags_df.drop(columns=["genres"])

    movies_ratings_tags_df["title"] = (
        movies_ratings_tags_df["title"].astype("category").cat.codes
    )  # 为电影标题创建唯一的整数编码
    return movies_ratings_tags_df


def link_df(a, b, on, how):
    """
    将a和b两个DataFrame根据on和how规则合并起来
    :param a: 要合并的第一个df
    :param b: 要合并的第二个df
    :param on: 根据什么值合并
    :param how: 怎么合并
    :return: 合并完的df
    """
    ans = pd.merge(a, b, on=on, how=how)
    return ans
