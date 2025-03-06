import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    """
    Load movie data from the u.item file.
    The dataset contains movie information, including genres.
    last 19 columns are genres
    """
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
    """
    Extract movie genres as features for similarity calculation.
    """
    genre_columns = columns[5:]
    features = movies[genre_columns]
    return features, genre_columns


def build_similarity_matrix(features):
    """
    Compute the cosine similarity matrix based on movie genres.
    """
    similarity_matrix = cosine_similarity(features)
    return similarity_matrix


def get_recommendations(input_titles, movies, similarity_matrix, genre_columns, top_n=10):
    """
    Given a list of input movies, return up to 10 recommended movies
    based on genre similarity.
    """
    input_indices = []
    for title in input_titles:
        match = movies[movies['title'] == title]
        if not match.empty:
            input_indices.append(match.index[0])

    if not input_indices:
        return []

    # Compute the average similarity across selected movies
    avg_similarity = similarity_matrix[input_indices].mean(axis=0)

    # Exclude input movies from recommendations
    for idx in input_indices:
        avg_similarity[idx] = -1

    # Get top recommended movies
    top_indices = avg_similarity.argsort()[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        if avg_similarity[idx] > 0:
            # Identify common genres
            input_genres = movies.iloc[input_indices][genre_columns].sum(axis=0)
            movie_genres = movies.iloc[idx][genre_columns]
            common_genres = movie_genres[movie_genres > 0].index.tolist()

            recommendations.append({
                'title': movies.iloc[idx]['title'],
                'similarity': avg_similarity[idx],
                'reason': f"Common genres: {', '.join(common_genres)}"
            })

    return recommendations


def main():
    """
    Main function to run the recommendation engine.
    """
    print("Loading movie data...")
    movies, columns = load_data()
    features, genre_columns = prepare_features(movies, columns)
    similarity_matrix = build_similarity_matrix(features)

    # Get user input
    input_titles = []
    print("Please enter up to 5 movies you like (press Enter to finish):")
    while len(input_titles) < 5:
        title = input(f"{len(input_titles) + 1}. Movie Name (Include year): ")
        if not title:
            break
        input_titles.append(title)

    # Generate recommendations
    recommendations = get_recommendations(input_titles, movies, similarity_matrix, genre_columns)

    # Display results
    print("\nRecommended Movies:")
    if not recommendations:
        print("No recommendations found.")
    else:
        for i, rec in enumerate(recommendations[:10], 1):
            print(f"{i}. {rec['title']}")
            print(f"   Reason: {rec['reason']}")
            print(f"   Similarity Score: {rec['similarity']:.2f}\n")


if __name__ == "__main__":
    main()

    """
    ### Explanation of the Code

    #### What It Does:
    1. **Loads Movie Data**: Reads the `u.item` file from the MovieLens dataset and extracts movie titles and genres.
    2. **Processes Features**: Uses movie genres as features to compute similarity.
    3. **Computes Similarity Matrix**: Uses cosine similarity to compare movies based on genre.
    4. **Takes User Input**: Allows the user to input up to 5 favorite movies, if they want stop, just press Enter to finish.
    5. **Generates Recommendations**: Suggests up to 10 movies similar to the input ones, along with explanations.

    #### What It Doesn’t Do:
    - It **does not** consider user ratings—recommendations are purely based on genre similarity.
    - It **does not** factor in popularity or recent trends.
    - It **does not** handle typos in movie names; users need to enter exact titles.
    - It **cannot** recommend movies that are not in the dataset.
    - It **cannot** detect or correct movie titles that are not in the dataset; if a movie name is entered incorrectly, no recommendations will be provided.

    #### Algorithm Used:
    - **Cosine Similarity**: Measures the similarity between movies based on genre vectors.

    #### Assumptions:
    - Movies with similar genres have a high probability of being liked together.
    - The dataset is clean and free from missing genre information.
    - Users input exact movie titles from the dataset.
    """
