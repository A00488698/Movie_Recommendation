
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    # Load movie data
    columns = [
        'movie_id', 'title', 'release_date', 'video_release', 'imdb_url',
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    movies = pd.read_csv(
        'ml-100k/u.item',
        sep='|',
        encoding='latin-1',
        header=None,
        names=columns
    )
    return movies, columns


def prepare_features(movies, columns):
    # extract feature 19 columns
    genre_columns = columns[5:]
    features = movies[genre_columns]
    return features, genre_columns


def build_similarity_matrix(features):
    # calculate cosine_similarity
    similarity_matrix = cosine_similarity(features)
    return similarity_matrix


def get_recommendations(input_titles, movies, similarity_matrix, genre_columns, top_n=10):
    # 获取输入电影的索引
    input_indices = []
    for title in input_titles:
        match = movies[movies['title'] == title]
        if not match.empty:
            input_indices.append(match.index[0])

    if not input_indices:
        return []

    # 计算平均相似度
    avg_similarity = similarity_matrix[input_indices].mean(axis=0)

    # 排除已输入的电影
    for idx in input_indices:
        avg_similarity[idx] = -1

    # 获取最相似的电影索引
    top_indices = avg_similarity.argsort()[::-1][:top_n]

    # 生成推荐结果和理由
    recommendations = []
    for idx in top_indices:
        if avg_similarity[idx] > 0:
            # 寻找共同类型
            input_genres = movies.iloc[input_indices][genre_columns].sum(axis=0)
            movie_genres = movies.iloc[idx][genre_columns]
            common_genres = movie_genres[movie_genres > 0].index.tolist()

            recommendations.append({
                'title': movies.iloc[idx]['title'],
                'similarity': avg_similarity[idx],
                'reason': f"Same feature：{', '.join(common_genres)}"
            })

    return recommendations


def main():
    # 加载数据
    print("Loading...")
    movies, columns = load_data()
    features, genre_columns = prepare_features(movies, columns)
    similarity_matrix = build_similarity_matrix(features)

    # 获取用户输入
    input_titles = []
    print(f'"Please input your favourite movies（up to 5 movies,"Enter to end "）："')
    while len(input_titles) < 5:
        title = input(f"{len(input_titles) + 1}. Movie Name（Include year）：")
        if not title:
            break
        input_titles.append(title)

    # 获取推荐
    recommendations = get_recommendations(input_titles, movies, similarity_matrix, genre_columns)

    # 显示结果
    print("\nRecommendations Results:：")
    if not recommendations:
        print("There is no recommendations.")
    else:
        for i, rec in enumerate(recommendations[:10], 1):
            print(f"{i}. {rec['title']}")
            print(f"   recommend reason：{rec['reason']}")
            print(f"   similarity：{rec['similarity']:.2f}\n")


if __name__ == "__main__":
    main()