# Movie Recommendation System UI

## Run locally
1. Install Python 3.10+.
2. Put `app.py`, `requirements.txt`, and `tmdb_5000_movies.csv` in the same folder.
3. Open a terminal in that folder and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. A browser tab will open. Upload `tmdb_5000_movies.csv` in the app and choose a movie.

## Deploy online (optional)
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Upload the dataset to the repository only if your course rules allow it and you are comfortable making it public. Otherwise, use the app's file uploader.
3. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the main file.
4. Open the app link and upload the CSV if prompted.

The app expects the TMDB 5000 dataset columns `title`, `genres`, `keywords`, and `overview`.
