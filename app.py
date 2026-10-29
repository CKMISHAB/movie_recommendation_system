import ast
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Choose a movie and discover other movies with similar content.")

@st.cache_data
def load_movies(path):
    df = pd.read_csv(path)
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
                return [str(item.get("name", "")) for item in items if isinstance(item, dict) and item.get("name")]
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
    df = df.dropna(subset=["title"]).drop_duplicates(subset=["title"]).reset_index(drop=True)
    return df

@st.cache_resource
def build_similarity(tags):
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(tags)
    return cosine_similarity(vectors)

uploaded = st.file_uploader("Upload tmdb_5000_movies.csv", type=["csv"])
dataset_path = None

if uploaded is not None:
    try:
        movies = load_movies(uploaded)
    except Exception as e:
        st.error(f"Could not read this dataset: {e}")
        st.stop()
else:
    st.info("Upload your **tmdb_5000_movies.csv** file above to start.")
    st.stop()

if movies.empty:
    st.error("The dataset has no usable movie rows.")
    st.stop()

with st.spinner("Preparing movie similarity data..."):
    similarity = build_similarity(movies["tags"].tolist())

movie_titles = movies["title"].tolist()
selected_title = st.selectbox("Select a movie", movie_titles)

num_recommendations = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("✨ Recommend movies", type="primary"):
    selected_index = movies.index[movies["title"] == selected_title][0]
    scores = list(enumerate(similarity[selected_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    recommendations = [(idx, score) for idx, score in scores if idx != selected_index][:num_recommendations]

    st.subheader(f"Movies similar to {selected_title}")
    if not recommendations:
        st.warning("No recommendations could be generated for this movie.")
    else:
        cols = st.columns(2)
        for position, (idx, score) in enumerate(recommendations):
            row = movies.iloc[idx]
            with cols[position % 2]:
                st.markdown(f"### {position + 1}. {row['title']}")
                genres = ", ".join(row["genres_clean"]) or "Genres unavailable"
                st.caption(f"Genres: {genres}")
                overview = row["overview_clean"].strip()
                if overview:
                    st.write(overview[:400] + ("..." if len(overview) > 400 else ""))
                st.progress(max(0.0, min(float(score), 1.0)))
                st.caption("Similarity score (relative): {:.3f}".format(float(score)))

st.divider()
st.caption("Built with Python, Pandas, Streamlit, CountVectorizer, and cosine similarity.")
