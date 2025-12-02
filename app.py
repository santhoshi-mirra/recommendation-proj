import streamlit as st
import pickle

#Load Data
movies = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
similarity = cosine_similarity(tfidf_matrix)

#Page Configuration
st.set_page_config(page_title="🎬 Movie Recommender System", layout="wide")

#Custom Styling
st.markdown("""
<style>
.stApp {
    background-color: #111;
    color: white;
    font-family: 'Poppins', sans-serif;
}
.stButton>button {
    background-color: #F97A5D;
    color: white;
    border-radius: 20px;
    padding: 10px 24px;
    border: none;
    font-weight: 600;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #ff9378;
}
h1, h2, h3, h4 {
    color: white;
}
.next-btn {
    display: flex;
    justify-content: center;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

#title
st.title("Movie Recommender System")
st.write("Find movies similar to your favorite one! Select a movie below 👇")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie:", movie_list)

#Recoms func
def recommend(movie, start=0):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = [movies.iloc[i[0]].title for i in distances[1:]]
    return recommendations[start:start+5]

#session
if "page" not in st.session_state:
    st.session_state.page = 0
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

#rec button
if st.button("Show Recommendations", key="show_btn"):
    st.session_state.page = 0
    st.session_state.recommendations = recommend(selected_movie)

#display
if st.session_state.recommendations:
    st.subheader("Recommended Movies:")
    names = st.session_state.recommendations
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        if idx < len(names):
            with col:
                st.write(f"**{names[idx]}**")

#nxt button
col1, col2, col3 = st.columns([3,1,3])
with col2:
    if st.button("Next →", key="next_btn"):
        st.session_state.page += 1
        start = st.session_state.page * 5
        st.session_state.recommendations = recommend(selected_movie, start)
