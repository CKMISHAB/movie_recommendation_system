import ast
import time
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("Compare two content-based models: CountVectorizer + Cosine Similarity and TF-IDF + KNN.")

@st.cache_data
def load_movies(file):
    df = pd.read_csv(file)
    required = {"title", "genres", "keywords", "overview"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    def parse_names(value):
        if pd.isna(value):
            return []
        try:
            items = ast.literal_eval(value) if isinstance(value, str) else value
            if isinstance(items, list):
                return [str(x.get("name", "")) for x in items
                        if isinstance(x, dict) and x.get("name")]
        except (ValueError, SyntaxError, TypeError):
            pass
        return []

    df["genres_clean"] = df["genres"].apply(parse_names)
    df["keywords_clean"] = df["keywords"].apply(parse_names)
    df["overview_clean"] = df["overview"].fillna("").astype(str)
    df["tags"] = (
        df["genres_clean"].apply(lambda x: " ".join(x)) + " " +
        df["keywords_clean"].apply(lambda x: " ".join(x)) + " " +
        df["overview_clean"]
    ).str.lower()
    return (df.dropna(subset=["title"])
              .drop_duplicates(subset=["title"]).reset_index(drop=True))

@st.cache_resource
def build_count_model(tags):
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(tags)
    return cosine_similarity(vectors)

@st.cache_resource
def build_tfidf_knn_model(tags):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(tags)
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(vectors)
    return knn, vectors

def count_recommendations(similarity, movie_index, count):
    scores = sorted(enumerate(similarity[movie_index]),
                    key=lambda item: item[1], reverse=True)
    return [(idx, float(score)) for idx, score in scores
            if idx != movie_index][:count]

def knn_recommendations(knn, vectors, movie_index, count):
    distances, indices = knn.kneighbors(
        vectors[movie_index], n_neighbors=min(count + 1, vectors.shape[0])
    )
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx != movie_index:
            results.append((int(idx), max(0.0, 1.0 - float(distance))))
        if len(results) == count:
            break
    return results

def show_results(movies, results, heading):
    st.subheader(heading)
    if not results:
        st.warning("No recommendations could be generated.")
        return
    for rank, (idx, score) in enumerate(results, 1):
        row = movies.iloc[idx]
        st.markdown(f"**{rank}. {row['title']}**")
        st.caption("Genres: " + (", ".join(row["genres_clean"]) or "Unavailable"))
        overview = row["overview_clean"].strip()
        if overview:
            st.write(overview[:300] + ("..." if len(overview) > 300 else ""))
        st.progress(min(1.0, max(0.0, score)))
        st.caption(f"Cosine-based similarity: {score:.3f}")

uploaded = st.file_uploader("Upload tmdb_5000_movies.csv", type=["csv"])
if uploaded is None:
    st.info("Upload your tmdb_5000_movies.csv file to start.")
    st.stop()

try:
    movies = load_movies(uploaded)
except Exception as error:
    st.error(f"Could not read this dataset: {error}")
    st.stop()

if len(movies) < 2:
    st.error("The dataset must contain at least two valid movies.")
    st.stop()

with st.spinner("Preparing both recommendation models..."):
    tags = tuple(movies["tags"].tolist())
    count_similarity = build_count_model(tags)
    tfidf_knn, tfidf_vectors = build_tfidf_knn_model(tags)

st.success(f"Both models are ready. Dataset contains {len(movies):,} movies.")
titles = movies["title"].tolist()
selected_title = st.selectbox("Select a movie", titles)
n = st.slider("Number of recommendations", 3, min(10, len(movies)-1),
              min(5, len(movies)-1))
selected_index = movies.index[movies["title"] == selected_title][0]

if st.button("✨ Compare / Recommend Movies", type="primary"):
    with st.spinner("Generating recommendations..."):
        start = time.perf_counter()
        count_results = count_recommendations(count_similarity, selected_index, n)
        count_time = time.perf_counter() - start

        start = time.perf_counter()
        knn_results = knn_recommendations(tfidf_knn, tfidf_vectors, selected_index, n)
        knn_time = time.perf_counter() - start

    st.session_state["comparison"] = {
        "title": selected_title,
        "count": count_results,
        "knn": knn_results,
        "count_time": count_time,
        "knn_time": knn_time,
    }

result = st.session_state.get("comparison")
if result is None or result["title"] != selected_title:
    st.info("Choose a movie and click the button to compare both models.")
else:
    tab_compare, tab_count, tab_knn = st.tabs(
        ["Compare Both", "Model 1: CountVectorizer", "Model 2: TF-IDF + KNN"]
    )
    with tab_compare:
        st.subheader(f"Recommendations for {selected_title}")
        left, right = st.columns(2)
        with left:
            show_results(movies, result["count"],
                         "Model 1 — CountVectorizer + Cosine Similarity")
        with right:
            show_results(movies, result["knn"],
                         "Model 2 — TF-IDF + K-Nearest Neighbors")
        count_titles = {movies.iloc[i]["title"] for i, _ in result["count"]}
        knn_titles = {movies.iloc[i]["title"] for i, _ in result["knn"]}
        common = sorted(count_titles & knn_titles)
        st.divider()
        st.subheader("Comparison")
        st.write(f"Common recommendations: **{len(common)}** out of {n}.")
        st.write(", ".join(common) if common else "No common titles in these lists.")
        st.write("Recommendation retrieval time (seconds):")
        st.dataframe(pd.DataFrame({
            "Model": ["CountVectorizer + Cosine Similarity", "TF-IDF + KNN"],
            "Time (seconds)": [result["count_time"], result["knn_time"]]
        }), use_container_width=True, hide_index=True)
        st.caption("Timing is for retrieving recommendations for this movie, not model-building time. It depends on the computer and dataset.")
    with tab_count:
        show_results(movies, result["count"],
                     "Model 1 — CountVectorizer + Cosine Similarity")
    with tab_knn:
        show_results(movies, result["knn"],
                     "Model 2 — TF-IDF + K-Nearest Neighbors")

st.divider()
st.caption("Built with Python, Pandas, Streamlit, CountVectorizer, TF-IDF, Cosine Similarity, and KNN.")

