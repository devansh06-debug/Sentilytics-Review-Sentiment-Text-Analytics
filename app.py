"""
Web app (Streamlit) for the Review Sentiment project.
Run with:  streamlit run app.py
Tabs: Predict | Data Analytics | NLP Explorer
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import analytics, data_gen
from src.nlp import clean_text, load_binary_dataset, preprocess, remove_stopwords, stem, tokenize

MODEL_PATH = Path("outputs/lstm_model.pt")

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="wide")


# ---------- make sure data exists ----------
@st.cache_data
def get_clean_data() -> pd.DataFrame:
    if not analytics.CLEAN_PATH.exists():
        if not analytics.DATA_PATH.exists():
            data_gen.main()
        analytics.CLEAN_PATH.parent.mkdir(exist_ok=True)
        analytics.clean(analytics.load_data()).to_csv(analytics.CLEAN_PATH, index=False)
    return pd.read_csv(analytics.CLEAN_PATH)


# ---------- NLP model (trained on the fly, cached) ----------
@st.cache_resource
def get_tfidf_model():
    get_clean_data()
    df = load_binary_dataset()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(df["processed"])
    clf = LogisticRegression(max_iter=1000).fit(X, df["label"])
    return vec, clf


# ---------- Deep learning model (optional: needs torch + trained file) ----------
@st.cache_resource
def get_lstm_model():
    if not MODEL_PATH.exists():
        return None, "Train it first with `python -m src.deep_learning`."
    try:
        import torch
        from src.deep_learning import SentimentLSTM
    except ImportError:
        return None, "PyTorch is not installed."
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    model = SentimentLSTM(len(vocab))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (model, vocab), None


def lstm_probability(model_and_vocab, text: str) -> float:
    import torch
    from src.deep_learning import encode
    model, vocab = model_and_vocab
    x = torch.tensor([encode(text, vocab)], dtype=torch.long)
    with torch.no_grad():
        return torch.sigmoid(model(x)).item()


# ================= UI =================
st.title("💬 Customer Review Sentiment Analyzer")
st.caption("Data Analytics + NLP + Deep Learning in one app")

tab_predict, tab_data, tab_nlp = st.tabs(["🔮 Predict", "📊 Data Analytics", "🧠 NLP Explorer"])

# ---------- Tab 1: Predict ----------
with tab_predict:
    examples = {
        "Positive example": "I absolutely love this laptop, excellent quality and fast delivery!",
        "Negative example": "Terrible product, it stopped working after two days. I want a refund.",
        "Custom": "",
    }
    choice = st.selectbox("Pick an example or write your own", list(examples))
    text = st.text_area("Review text", value=examples[choice], height=120)

    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            col1, col2 = st.columns(2)

            vec, clf = get_tfidf_model()
            p_tfidf = clf.predict_proba(vec.transform([preprocess(text)]))[0][1]
            with col1:
                st.subheader("NLP model (TF-IDF + Logistic Regression)")
                st.metric("Prediction", "Positive 😊" if p_tfidf > 0.5 else "Negative 😞")
                st.progress(float(p_tfidf), text=f"Positive probability: {p_tfidf:.2f}")

            lstm, err = get_lstm_model()
            with col2:
                st.subheader("Deep learning model (BiLSTM)")
                if lstm is None:
                    st.info(err)
                else:
                    p_lstm = lstm_probability(lstm, text)
                    st.metric("Prediction", "Positive 😊" if p_lstm > 0.5 else "Negative 😞")
                    st.progress(float(p_lstm), text=f"Positive probability: {p_lstm:.2f}")

# ---------- Tab 2: Data analytics ----------
with tab_data:
    df = get_clean_data()
    categories = st.multiselect("Filter by category", sorted(df["category"].unique()),
                                default=sorted(df["category"].unique()))
    view = df[df["category"].isin(categories)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reviews", len(view))
    m2.metric("Average rating", f"{view['rating'].mean():.2f}")
    m3.metric("Avg review length (words)", f"{view['review_length'].mean():.1f}")
    m4.metric("% positive", f"{(view['sentiment'] == 'positive').mean() * 100:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rating distribution")
        st.bar_chart(view["rating"].value_counts().sort_index())
    with c2:
        st.subheader("Average rating by category")
        st.bar_chart(view.groupby("category")["rating"].mean())

    st.subheader("Review length distribution")
    st.bar_chart(view["review_length"].value_counts().sort_index())

    st.subheader("Correlation")
    st.dataframe(view[["rating", "helpful_votes", "review_length"]].corr().round(2))

    st.subheader("Sample data")
    st.dataframe(view.head(20), use_container_width=True)

# ---------- Tab 3: NLP explorer ----------
with tab_nlp:
    st.write("See how raw text is transformed step by step before a model sees it.")
    raw = st.text_input("Type a sentence",
                        "THE HEADPHONES were amazing!!! Check http://spam.example.com <br>")
    cleaned = clean_text(raw)
    tokens = tokenize(cleaned)
    no_stop = remove_stopwords(tokens)
    stemmed = [stem(t) for t in no_stop]

    st.markdown("**1. Cleaned text** (lowercase, no URLs/HTML/punctuation)")
    st.code(cleaned)
    st.markdown("**2. Tokens**")
    st.code(str(tokens))
    st.markdown("**3. After stop-word removal**")
    st.code(str(no_stop))
    st.markdown("**4. After stemming**")
    st.code(str(stemmed))

    st.subheader("Most influential words (from the trained model)")
    vec, clf = get_tfidf_model()
    import numpy as np
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    a, b = st.columns(2)
    a.markdown("**Negative**"); a.write(list(names[order[:10]]))
    b.markdown("**Positive**"); b.write(list(names[order[-10:]][::-1]))
