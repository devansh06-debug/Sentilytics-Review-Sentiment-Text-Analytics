# Customer Review Sentiment Analysis
A beginner-friendly project that combines **Data Analytics**, **NLP** and **Deep Learning**
in one pipeline: understand the data -> process the text -> train a neural network.

## Quick start
```bash
pip install -r requirements.txt
python run_all.py          # runs everything
# or step by step:
python -m src.data_gen
python -m src.analytics
python -m src.nlp
python -m src.deep_learning
```
Results (plots, trained model) are saved in `outputs/`.

## Web app
```bash
python -m src.deep_learning     # optional: trains the LSTM so the app can use it
streamlit run app.py            # opens http://localhost:8501
```
The app has 3 tabs: **Predict** (type a review, see NLP and LSTM predictions),
**Data Analytics** (interactive filters and charts) and **NLP Explorer**
(see each text-processing step, plus the most positive/negative words).
It works even without the LSTM; that panel just shows a hint until the model is trained.

## What each part covers

| Part | File | Topics you learn |
|------|------|------------------|
| Data Analytics | `src/analytics.py` | loading CSV, inspecting data, missing values, duplicates, cleaning, feature engineering, group-by, correlation, charts |
| NLP | `src/nlp.py` | lowercasing, removing URLs/HTML, tokenization, stop words (keeping negations), stemming, TF-IDF, n-grams, Logistic Regression baseline, precision/recall/F1, most influential words |
| Deep Learning | `src/deep_learning.py` | vocabulary, padding, embeddings, bidirectional LSTM, Dataset/DataLoader, loss + optimizer, train/val/test split, dropout, early stopping, saving model, inference on new text |

## Project structure
```
review_sentiment_project/
├── data/               # reviews.csv (raw), reviews_clean.csv
├── outputs/            # eda_plots.png, training_curves.png, lstm_model.pt
├── src/
│   ├── data_gen.py     # creates synthetic review data
│   ├── analytics.py
│   ├── nlp.py
│   └── deep_learning.py
├── app.py              # Streamlit web app
├── run_all.py
└── requirements.txt
```

## Using real data
Replace `data/reviews.csv` (skip `data_gen`) with your own file containing:
`review_text, rating, category, helpful_votes`.

## Ideas to extend
- Compare TF-IDF + Logistic Regression vs LSTM results
- Try pre-trained embeddings (GloVe) or a transformer (Hugging Face `distilbert`)
- Add a 3-class model (include neutral), word clouds, or a Streamlit app
