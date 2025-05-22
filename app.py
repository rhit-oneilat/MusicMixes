import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Song Genre Prediction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning visual design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Main Header with advanced styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }

    .main-header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        z-index: 2;
        position: relative;
    }

    .main-header p {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
        z-index: 2;
        position: relative;
    }

    /* Enhanced Song Cards with fixed alignment */
    .song-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .song-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }

    .song-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }

    /* Glass morphism effect for feature boxes with fixed height */
    .feature-box {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .feature-box:hover {
        background: rgba(255, 255, 255, 0.9);
        transform: translateY(-2px);
    }

    /* Enhanced Genre Prediction Cards */
    .genre-prediction {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .genre-prediction::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .genre-prediction h4, .genre-prediction h2 {
        z-index: 2;
        position: relative;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        font-family: 'Poppins', sans-serif;
    }

    /* Modern Stats Cards with fixed alignment */
    .stats-card {
        background: linear-gradient(145deg, #ffffff 0%, #f0f2f6 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.8rem 0.2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .stats-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    .stats-card:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 25px rgba(0,0,0,0.15);
    }

    .stats-card h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }

    /* Enhanced Search Result Items with fixed height */
    .search-result-item {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .search-result-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s;
    }

    .search-result-item:hover::before {
        left: 100%;
    }

    .search-result-item:hover {
        transform: translateX(8px) translateY(-2px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        border-left-width: 6px;
    }

    /* Animated Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 10px;
    }

    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Sidebar Enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9ff 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #ffffff 0%, #f0f2f6 100%);
        border-radius: 12px;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    /* Welcome Section Styling with Modern Typography */
    .welcome-section {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
        padding: 4rem 2rem;
        border-radius: 25px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
        font-family: 'Poppins', sans-serif;
    }

    .welcome-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        line-height: 1.2;
    }

    .welcome-subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.4rem;
        color: #6c757d;
        line-height: 1.6;
        max-width: 700px;
        margin: 0 auto 3rem auto;
    }

    /* Feature Cards Grid with Equal Heights */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 3rem 0;
    }

    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f0f8ff 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.2rem;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
    }

    .feature-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.3rem;
        color: #2c3e50;
        margin-bottom: 0.8rem;
    }

    .feature-description {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 0.95rem;
        color: #6c757d;
        line-height: 1.5;
    }

    /* Getting Started Section */
    .getting-started {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 3rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }

    .getting-started::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        animation: float 8s ease-in-out infinite;
    }

    .getting-started-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        z-index: 2;
        position: relative;
    }

    .steps-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
        z-index: 2;
        position: relative;
    }

    .step-card {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 1.8rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .step-card:hover {
        background: rgba(255,255,255,0.25);
        transform: translateY(-5px);
    }

    .step-number {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }

    .step-description {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        opacity: 0.9;
        line-height: 1.4;
    }

    /* Match Type Badges */
    .match-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }

    /* Confidence Indicator */
    .confidence-indicator {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 10px 15px;
        border-radius: 10px;
        font-size: 0.95rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        color: #495057;
        border-left: 4px solid #667eea;
    }

    /* Fixed height containers for search results */
    .search-results-container {
        display: grid;
        gap: 1rem;
    }

    .search-result-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr;
        gap: 1rem;
        align-items: stretch;
        min-height: 140px;
    }

    .result-main-info {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .result-side-info {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 100%;
    }

    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 2s ease-in-out infinite;
    }

    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 12px;
        border: none;
    }

    .stError {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        border-radius: 12px;
        border: none;
    }

    .stWarning {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        border-radius: 12px;
        border: none;
    }

    .stInfo {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        border-radius: 12px;
        border: none;
    }

    /* Metric Cards */
    .css-1xarl3l {
        background: linear-gradient(145deg, #ffffff 0%, #f0f2f6 100%);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .welcome-title {
            font-size: 2.5rem;
        }

        .welcome-subtitle {
            font-size: 1.1rem;
        }

        .main-header h1 {
            font-size: 2rem;
        }

        .main-header p {
            font-size: 1rem;
        }

        .song-card, .genre-prediction {
            padding: 1.5rem;
        }

        .feature-grid {
            grid-template-columns: 1fr;
        }

        .steps-grid {
            grid-template-columns: 1fr;
        }

        .search-result-row {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("demo2.csv")
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        df['title'] = df['track_name'].fillna("")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

# Build TF-IDF index on titles
@st.cache_resource
def build_title_index(titles):
    # More sensitive vectorizer settings
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Include trigrams for better matching
        min_df=1,  # Don't ignore rare terms
        max_df=0.8,  # Ignore very common terms
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b\w+\b',  # Better word boundaries
        sublinear_tf=True  # Use sublinear TF scaling
    )
    tfidf_matrix = vectorizer.fit_transform(titles)
    return vectorizer, tfidf_matrix

# Enhanced search with multiple methods
def enhanced_search(query, vectorizer, tfidf_matrix, df, min_similarity=0.1, top_n=50):
    if not query or vectorizer is None or tfidf_matrix is None:
        return pd.DataFrame()

    query_lower = query.lower().strip()
    results_list = []

    # Method 1: Exact substring matching (highest priority)
    exact_matches = df[df['title'].str.lower().str.contains(query_lower, na=False, regex=False)]
    if not exact_matches.empty:
        exact_matches = exact_matches.copy()
        exact_matches['similarity_score'] = 1.0
        exact_matches['match_type'] = 'Exact Match'
        results_list.append(exact_matches)

    # Method 2: Word-by-word matching
    query_words = query_lower.split()
    if len(query_words) > 1:
        word_pattern = '|'.join([f'\\b{word}\\b' for word in query_words])
        word_matches = df[df['title'].str.lower().str.contains(word_pattern, na=False, regex=True)]
        if not word_matches.empty:
            word_matches = word_matches.copy()
            # Calculate match score based on how many words match
            def calculate_word_score(title):
                title_lower = str(title).lower()
                matches = sum(1 for word in query_words if word in title_lower)
                return 0.8 * (matches / len(query_words))

            word_matches['similarity_score'] = word_matches['title'].apply(calculate_word_score)
            word_matches['match_type'] = 'Word Match'
            word_matches = word_matches[word_matches['similarity_score'] > 0.3]
            results_list.append(word_matches)

    # Method 3: TF-IDF similarity (for fuzzy matching)
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Only include results above minimum similarity threshold
    valid_indices = np.where(similarity >= min_similarity)[0]
    if len(valid_indices) > 0:
        tfidf_matches = df.iloc[valid_indices].copy()
        tfidf_matches['similarity_score'] = similarity[valid_indices] * 0.6  # Lower weight for fuzzy matches
        tfidf_matches['match_type'] = 'Similar'
        results_list.append(tfidf_matches)

    # Combine all results
    if not results_list:
        return pd.DataFrame()

    combined_results = pd.concat(results_list, ignore_index=True)

    # Remove duplicates, keeping the best score for each song
    combined_results = combined_results.sort_values(['title', 'similarity_score'], ascending=[True, False])
    combined_results = combined_results.drop_duplicates(subset=['title'], keep='first')

    # Sort by similarity score and match type priority
    match_type_priority = {'Exact Match': 3, 'Word Match': 2, 'Similar': 1}
    combined_results['priority'] = combined_results['match_type'].map(match_type_priority)
    combined_results = combined_results.sort_values(['priority', 'similarity_score'], ascending=[False, False])

    return combined_results.head(top_n)

# Artist search function
def search_by_artist(query, df, min_similarity=0.1):
    if not query:
        return pd.DataFrame()

    query_lower = query.lower().strip()

    # Exact artist match
    exact_artist = df[df['artists'].str.lower().str.contains(query_lower, na=False, regex=False)]
    if not exact_artist.empty:
        exact_artist = exact_artist.copy()
        exact_artist['similarity_score'] = 1.0
        exact_artist['match_type'] = 'Artist Match'
        return exact_artist.head(50)

    return pd.DataFrame()

# Create audio features radar chart
def create_audio_features_chart(features_dict):
    features = list(features_dict.keys())
    values = list(features_dict.values())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=features,
        fill='toself',
        name='Audio Features',
        line_color='rgb(102, 126, 234)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        height=400,
        title="Audio Features Profile"
    )
    return fig

# Create confidence gauge
def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# Load data
df = load_data()

if not df.empty and 'title' in df.columns:
    vectorizer, tfidf_matrix = build_title_index(df['title'])

    # Sidebar
    with st.sidebar:
        st.header("🔍 Search & Filter")

        # Search options
        search_type = st.radio(
            "Search by:",
            ["Song Title", "Artist", "Both"],
            index=0
        )

        # Search input
        if search_type == "Song Title":
            query = st.text_input("🎤 Search by Song Title", placeholder="Enter song name...")
            artist_query = ""
        elif search_type == "Artist":
            query = ""
            artist_query = st.text_input("👤 Search by Artist", placeholder="Enter artist name...")
        else:  # Both
            query = st.text_input("🎤 Song Title", placeholder="Enter song name...")
            artist_query = st.text_input("👤 Artist", placeholder="Enter artist name...")

        # Search sensitivity settings
        st.markdown("---")
        st.subheader("🎯 Search Settings")

        sensitivity = st.select_slider(
            "Search Sensitivity",
            options=["Very Strict", "Strict", "Moderate", "Relaxed"],
            value="Strict",
            help="Higher sensitivity = more precise matches"
        )

        # Convert sensitivity to threshold
        sensitivity_map = {
            "Very Strict": 0.3,
            "Strict": 0.2,
            "Moderate": 0.1,
            "Relaxed": 0.05
        }
        min_similarity = sensitivity_map[sensitivity]

        max_results_sidebar = st.selectbox(
            "Maximum Results",
            [10, 20, 30, 50],
            index=1
        )

        # Show only exact matches option
        exact_only = st.checkbox("Show only exact/close matches", value=False)

        # Dataset statistics
        st.markdown("---")
        st.subheader("📊 Dataset Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <h3>{len(df):,}</h3>
                <p style="color: #6c757d; font-weight: 500;">Total Songs</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            unique_artists = df['artists'].nunique() if 'artists' in df.columns else 0
            st.markdown(f"""
            <div class="stats-card">
                <h3>{unique_artists:,}</h3>
                <p style="color: #6c757d; font-weight: 500;">Unique Artists</p>
            </div>
            """, unsafe_allow_html=True)

        # Genre distribution if available
        if 'actual_genre' in df.columns:
            st.markdown("---")
            st.subheader("🎭 Genre Distribution")
            genre_counts = df['actual_genre'].value_counts().head(10)
            fig_bar = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                color=genre_counts.values,
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(
                height=300,
                showlegend=False,
                xaxis_title="Count",
                yaxis_title="Genre"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Main content area
    if query or artist_query:
        if (query and len(query) < 2) or (artist_query and len(artist_query) < 2):
            st.warning("⚠️ Please enter at least 2 characters for search.")
        else:
            # Determine search approach
            results = pd.DataFrame()

            if search_type == "Song Title" and query:
                results = enhanced_search(query, vectorizer, tfidf_matrix, df, min_similarity, max_results_sidebar)
            elif search_type == "Artist" and artist_query:
                results = search_by_artist(artist_query, df, min_similarity)
            elif search_type == "Both":
                # Combine song and artist searches
                song_results = pd.DataFrame()
                artist_results = pd.DataFrame()

                if query:
                    song_results = enhanced_search(query, vectorizer, tfidf_matrix, df, min_similarity, max_results_sidebar//2)
                if artist_query:
                    artist_results = search_by_artist(artist_query, df, min_similarity)

                # If both searches, find intersection or union
                if not song_results.empty and not artist_results.empty:
                    # Find songs that match both criteria
                    intersection = song_results[song_results['title'].isin(artist_results['title'])]
                    if not intersection.empty:
                        results = intersection
                        st.info("🎯 Showing songs that match both title and artist criteria")
                    else:
                        # Show both results but clearly separated
                        results = pd.concat([song_results.head(max_results_sidebar//2),
                                           artist_results.head(max_results_sidebar//2)], ignore_index=True)
                        results = results.drop_duplicates(subset=['title'], keep='first')
                elif not song_results.empty:
                    results = song_results
                elif not artist_results.empty:
                    results = artist_results

            # Apply exact match filter if requested
            if exact_only and not results.empty:
                results = results[results['match_type'].isin(['Exact Match', 'Artist Match'])]

            # Results header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if not results.empty:
                    match_types = results['match_type'].value_counts()
                    match_summary = " | ".join([f"{count} {mtype}" for mtype, count in match_types.items()])
                    st.subheader(f"🎯 Found {len(results)} songs")
                    st.caption(f"Match types: {match_summary}")
                else:
                    st.subheader("🎯 Search Results: 0 songs found")

            with col2:
                if not results.empty:
                    # Filter by match type
                    available_types = results['match_type'].unique()
                    selected_types = st.multiselect(
                        "Filter by match type:",
                        available_types,
                        default=available_types,
                        key="match_filter"
                    )
                    if selected_types:
                        results = results[results['match_type'].isin(selected_types)]

            if results.empty:
                st.warning("🔍 No results found. Try:")
                st.markdown("""
                - Reducing search sensitivity
                - Using fewer/different keywords
                - Checking spelling
                - Searching by artist instead
                - Disabling 'exact matches only'
                """)
            else:
                # Display results in a more attractive format
                st.markdown("### 📝 Search Results")

                # Create tabs for different views
                tab1, tab2 = st.tabs(["🎵 Song List", "📊 Analysis"])

                with tab1:
                    # Display results with improved alignment
                    st.markdown('<div class="search-results-container">', unsafe_allow_html=True)

                    for idx, (_, song) in enumerate(results.head(max_results_sidebar).iterrows()):
                        similarity_pct = song['similarity_score'] * 100
                        match_type = song.get('match_type', 'Similar')

                        # Color coding for match types
                        match_colors = {
                            'Exact Match': '#28a745',  # Green
                            'Word Match': '#17a2b8',   # Blue
                            'Artist Match': '#6f42c1',  # Purple
                            'Similar': '#6c757d'       # Gray
                        }

                        match_color = match_colors.get(match_type, '#6c757d')

                        # Create a container for each result with fixed layout
                        with st.container():
                            st.markdown(f'<div class="search-result-row">', unsafe_allow_html=True)

                            col1, col2, col3 = st.columns([3, 1, 1])

                            with col1:
                                # Enhanced match badge styling
                                badge_style = f"background: {match_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 500;"

                                st.markdown(f"""
                                <div class="search-result-item" style="border-left-color: {match_color};">
                                    <div class="result-main-info">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                            <h4 style="margin: 0; color: #2c3e50; font-weight: 600; font-family: 'Poppins', sans-serif;">🎵 {song['title']}</h4>
                                            <span style="{badge_style}">
                                                {match_type}
                                            </span>
                                        </div>
                                        <p style="margin: 8px 0; color: #34495e; font-family: 'Inter', sans-serif;"><strong>👤 Artist:</strong> {song['artists']}</p>
                                        <div class="confidence-indicator">
                                            <strong>🎯 Match Confidence:</strong> {similarity_pct:.1f}%
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            with col2:
                                st.markdown('<div class="result-side-info">', unsafe_allow_html=True)
                                if 'album_name' in song and pd.notna(song['album_name']):
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(145deg, #f8f9ff 0%, #ffffff 100%);
                                                padding: 1rem; border-radius: 12px; text-align: center;
                                                border: 1px solid rgba(102, 126, 234, 0.1);
                                                min-height: 80px; display: flex; flex-direction: column; justify-content: center;">
                                        <small style="color: #6c757d; font-family: 'Inter', sans-serif;">💿 Album</small><br>
                                        <strong style="color: #495057; font-family: 'Poppins', sans-serif; font-size: 0.9rem;">{str(song['album_name'])[:25]}{'...' if len(str(song['album_name'])) > 25 else ''}</strong>
                                    </div>
                                    """, unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                            with col3:
                                st.markdown('<div class="result-side-info">', unsafe_allow_html=True)
                                # Enhanced button styling handled by CSS
                                if st.button(f"✨ View Details", key=f"view_{idx}", help="Click to see full song analysis", use_container_width=True):
                                    st.session_state['selected_song'] = song
                                    st.session_state['selected_index'] = idx
                                st.markdown('</div>', unsafe_allow_html=True)

                            st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                with tab2:
                    if len(results) > 0:
                        # Match type distribution
                        st.markdown("#### 🎯 Match Type Distribution")
                        match_counts = results['match_type'].value_counts()

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            fig_pie = px.pie(
                                values=match_counts.values,
                                names=match_counts.index,
                                title="Match Types"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with col2:
                            # Confidence distribution
                            st.markdown("#### 📊 Match Confidence")
                            fig_hist = px.histogram(
                                results,
                                x='similarity_score',
                                nbins=10,
                                title="Confidence Score Distribution",
                                labels={'similarity_score': 'Match Confidence', 'count': 'Number of Songs'}
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                        # Audio features comparison
                        audio_features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'valence']
                        available_features = [f for f in audio_features if f in results.columns]

                        if available_features:
                            st.markdown("#### 🎼 Audio Features Distribution")
                            feature_data = results[available_features].head(10)

                            fig_features = px.line_polar(
                                feature_data,
                                theta=available_features,
                                line_close=True,
                                title="Audio Features Comparison (Top 10 Results)"
                            )
                            st.plotly_chart(fig_features, use_container_width=True)

                # Detailed song view with enhanced styling
                if 'selected_song' in st.session_state:
                    st.markdown("---")

                    # Animated section header
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h2 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                                   font-size: 2.5rem; font-weight: 700; font-family: 'Poppins', sans-serif;">
                            🎤 Song Analysis Dashboard
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)

                    selected_song = st.session_state['selected_song']

                    # Enhanced song information card
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        album_info = f"<p style='margin: 8px 0; color: #6c757d; font-family: \"Inter\", sans-serif;'><strong>💿 Album:</strong> {selected_song['album_name']}</p>" if 'album_name' in selected_song and pd.notna(selected_song['album_name']) else ""
                        release_info = f"<p style='margin: 8px 0; color: #6c757d; font-family: \"Inter\", sans-serif;'><strong>📅 Release Date:</strong> {selected_song['release_date']}</p>" if 'release_date' in selected_song and pd.notna(selected_song['release_date']) else ""

                        st.markdown(f"""
                        <div class="song-card">
                            <h3 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.8rem; font-family: 'Poppins', sans-serif;">🎵 {selected_song['title']}</h3>
                            <p style="margin: 8px 0; color: #34495e; font-size: 1.1rem; font-family: 'Inter', sans-serif;"><strong>👤 Artist:</strong> {selected_song['artists']}</p>
                            {album_info}
                            {release_info}
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        # Enhanced metrics with better styling
                        if 'tempo' in selected_song and pd.notna(selected_song['tempo']):
                            st.markdown(f"""
                            <div class="feature-box">
                                <h4 style="color: #667eea; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif;">🎵 Tempo</h4>
                                <h2 style="color: #2c3e50; margin: 0; font-family: 'Poppins', sans-serif;">{selected_song['tempo']:.0f}</h2>
                                <small style="color: #6c757d; font-family: 'Inter', sans-serif;">BPM</small>
                            </div>
                            """, unsafe_allow_html=True)

                    # Audio features visualization
                    col1, col2 = st.columns([1, 1])

                    audio_features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'valence']
                    available_features = {f: selected_song[f] for f in audio_features if f in selected_song}

                    if available_features:
                        with col1:
                            st.markdown("#### 🎼 Audio Features")
                            radar_chart = create_audio_features_chart(available_features)
                            st.plotly_chart(radar_chart, use_container_width=True)

                        with col2:
                            st.markdown("""
                            <div style="background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
                                        padding: 1.5rem; border-radius: 15px; margin-top: 1rem;
                                        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
                                        border: 1px solid rgba(102, 126, 234, 0.1);">
                                <h4 style="color: #667eea; text-align: center; margin-bottom: 1rem; font-family: 'Poppins', sans-serif;">📊 Feature Values</h4>
                            </div>
                            """, unsafe_allow_html=True)

                            for feature, value in available_features.items():
                                # Create custom progress bar with better styling
                                progress_html = f"""
                                <div style="margin: 0.8rem 0;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                                        <span style="font-weight: 500; color: #2c3e50; font-family: 'Inter', sans-serif;">{feature.capitalize()}</span>
                                        <span style="font-weight: 600; color: #667eea; font-family: 'Poppins', sans-serif;">{value:.3f}</span>
                                    </div>
                                    <div style="background: #e9ecef; border-radius: 10px; height: 8px; overflow: hidden;">
                                        <div style="background: linear-gradient(90deg, #667eea, #764ba2);
                                                   width: {value*100}%; height: 100%; border-radius: 10px;
                                                   transition: width 0.5s ease;"></div>
                                    </div>
                                </div>
                                """
                                st.markdown(progress_html, unsafe_allow_html=True)

                    # Enhanced Genre prediction section
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h3 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                                   font-size: 2rem; font-weight: 600; font-family: 'Poppins', sans-serif;">
                            🎭 AI Genre Prediction
                        </h3>
                        <p style="color: #6c757d; font-size: 1.1rem; font-family: 'Inter', sans-serif;">Discover what our AI thinks about this song's genre</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Get prediction data
                    predicted_genre = selected_song.get('predicted_genre', 'Not available')
                    actual_genre = selected_song.get('actual_genre', 'Not available')
                    confidence = selected_song.get('confidence_interval', 0.0)

                    # Enhanced reveal button with session state
                    if 'show_prediction' not in st.session_state:
                        st.session_state['show_prediction'] = False

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if not st.session_state['show_prediction']:
                            st.markdown("""
                            <div style="text-align: center; padding: 2rem;
                                        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
                                        border-radius: 20px; margin: 1rem 0;
                                        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                                        border: 1px solid rgba(102, 126, 234, 0.1);">
                                <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
                                <h4 style="color: #2c3e50; margin-bottom: 1rem; font-family: 'Poppins', sans-serif;">Ready to reveal the AI's prediction?</h4>
                                <p style="color: #6c757d; font-family: 'Inter', sans-serif;">Click below to see how well our model performed!</p>
                            </div>
                            """, unsafe_allow_html=True)

                            if st.button("🚀 Reveal Genre Prediction", type="primary", use_container_width=True):
                                st.session_state['show_prediction'] = True
                                st.rerun()

                    if st.session_state['show_prediction']:
                        # Enhanced results display with animations
                        st.markdown("""
                        <div style="text-align: center; margin: 1rem 0;">
                            <div style="font-size: 3rem; animation: pulse 2s ease-in-out infinite;">🎉</div>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2, col3 = st.columns([1, 1, 1])

                        with col1:
                            st.markdown(f"""
                            <div class="genre-prediction">
                                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
                                <h4 style="margin-bottom: 0.5rem; font-size: 1.2rem;">AI Predicted</h4>
                                <h2 style="margin: 0; font-size: 1.8rem;">{predicted_genre}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                            <div class="genre-prediction">
                                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">✅</div>
                                <h4 style="margin-bottom: 0.5rem; font-size: 1.2rem;">Actual Genre</h4>
                                <h2 style="margin: 0; font-size: 1.8rem;">{actual_genre}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            # Enhanced confidence display
                            confidence_pct = confidence * 100
                            confidence_color = "#28a745" if confidence_pct >= 80 else "#ffc107" if confidence_pct >= 60 else "#dc3545"

                            st.markdown(f"""
                            <div class="genre-prediction">
                                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                                <h4 style="margin-bottom: 0.5rem; font-size: 1.2rem;">Confidence</h4>
                                <h2 style="margin: 0; font-size: 1.8rem;">{confidence_pct:.1f}%</h2>
                                <div style="background: rgba(255,255,255,0.3); border-radius: 10px; height: 8px; margin-top: 1rem; overflow: hidden;">
                                    <div style="background: {confidence_color}; width: {confidence_pct}%; height: 100%; border-radius: 10px; transition: width 1s ease;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Enhanced accuracy indicator
                        st.markdown("<br>", unsafe_allow_html=True)

                        if predicted_genre.lower() == actual_genre.lower():
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                                        color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                                        box-shadow: 0 10px 25px rgba(40, 167, 69, 0.3); margin: 1rem 0;">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎉</div>
                                <h3 style="margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); font-family: 'Poppins', sans-serif;">Perfect Match!</h3>
                                <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-family: 'Inter', sans-serif;">The AI successfully identified the correct genre</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
                                        color: white; padding: 1.5rem; border-radius: 15px; text-align: center;
                                        box-shadow: 0 10px 25px rgba(220, 53, 69, 0.3); margin: 1rem 0;">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                                <h3 style="margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); font-family: 'Poppins', sans-serif;">Close, but not quite!</h3>
                                <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-family: 'Inter', sans-serif;">The AI's prediction differs from the actual genre</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # reset button
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            if st.button("🔄 Hide Prediction", use_container_width=True):
                                st.session_state['show_prediction'] = False
                                st.rerun()

    else:
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; min-height: 50vh; margin: 2rem 0;">
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                       border-radius: 20px;
                       padding: 3rem 4rem;
                       box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                       border: 1px solid rgba(220, 53, 69, 0.1);
                       text-align: center;
                       max-width: 900px;
                       width: 90%;">
                <h2 style="background: linear-gradient(135deg, #dc3545 0%, #7600bc 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                           font-size: 3.5rem; font-weight: 700; font-family: 'Poppins', sans-serif;
                           margin-bottom: 1rem; line-height: 1.2;">
                    Welcome to MusicMixes
                </h2>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Enhanced error message with better styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
                color: white; padding: 2rem; border-radius: 20px; text-align: center; margin: 2rem 0;
                box-shadow: 0 15px 35px rgba(220, 53, 69, 0.3);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">⚠️</div>
        <h2 style="margin-bottom: 1rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); font-family: 'Poppins', sans-serif;">Data Loading Error</h2>
        <p style="opacity: 0.9; margin-bottom: 1rem; font-family: 'Inter', sans-serif;">Unable to load song data. Please check your CSV file configuration.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
                padding: 2rem; border-radius: 15px; margin: 1rem 0;
                box-shadow: 0 8px 20px rgba(0,0,0,0.1); border: 1px solid rgba(102, 126, 234, 0.1);">
        <h4 style="color: #2c3e50; margin-bottom: 1rem; font-family: 'Poppins', sans-serif;">📋 Required CSV Structure</h4>
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; font-family: 'Inter', monospace;">
            <strong>Required columns:</strong><br>
            • <code>track_name</code> or <code>title</code> - Song titles<br>
            • <code>artists</code> - Artist names<br><br>

            <strong>Optional columns:</strong><br>
            • Audio features: <code>danceability</code>, <code>energy</code>, <code>acousticness</code>, etc.<br>
            • Genre info: <code>predicted_genre</code>, <code>actual_genre</code>, <code>confidence_interval</code><br>
            • Metadata: <code>album_name</code>, <code>release_date</code>, <code>tempo</code>, <code>key</code>
        </div>
    </div>
    """)
