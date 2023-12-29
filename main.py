# 最后编辑：
# 晋晨曦 2023.12.19 23:38
# qq：2950171570
# email：Jin0714@outlook.com  回复随缘

import read as r
import data_process as dp
import soluation as s
import matplotlib


def main():
    # 设置字体：
    matplotlib.rcParams["font.family"] = "SimHei"  # 例如使用 "SimHei" 字体

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
    s.solve_ncf(df, 32, 50, 0.1, 0.00095)


if __name__ == "__main__":
    main()
