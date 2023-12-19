import read as r
import data_process as dp
import soluation as s


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
    s.solve_ncf(df, 5, 32, 50, 0.1, 108)


if __name__ == "__main__":
    main()
