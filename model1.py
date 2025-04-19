import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

@st.cache_resource
def train_sentiment_model():
    model = LogisticRegression()
    vectorizer = TfidfVectorizer(max_features=5000)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    clean_text = preprocess(text)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]
    return prediction

def extract_problems(negative_reviews):
    problem_keywords = {
        'bad', 'poor', 'broken', 'faulty', 'defective', 'issue',
        'problem', 'slow', 'crash', 'error', 'fail', 'stopped',
        'damaged', 'wrong', 'missing', 'waste', 'worst', 'unusable',
        'freeze', 'lag', 'noisy', 'overheat'
    }
    all_words = []
    for review in negative_reviews:
        words = word_tokenize(preprocess(review))
        problem_words = [word for word in words if word in problem_keywords]
        all_words.extend(problem_words)

    word_counts = Counter(all_words)
    common_problems = word_counts.most_common(3)
    return common_problems

st.title("📊 AI-Driven Sentiment Analysis Tool")
st.write("Upload a CSV file with reviews to analyze sentiment and identify product issues.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        st.subheader("📄 Original Data")
        st.write(df)

        review_column = None
        for col in ['Review', 'Summary', 'Text']:
            if col in df.columns:
                review_column = col
                break

        if not review_column:
            st.error("⚠ No review text column found. Expected 'Review', 'Summary', or 'Text'.")
        else:
            df['clean_text'] = df[review_column].apply(preprocess)

            model, vectorizer = train_sentiment_model()

            if 'Rate' in df.columns:
                df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
                df = df.dropna(subset=['Rate'])
                df['Rate'] = df['Rate'].astype(int)

                df['sentiment'] = df['Rate'].apply(
                    lambda x: 'positive' if x >= 4 else ('neutral' if x == 3 else 'negative'))
            else:
                df['sentiment'] = df[review_column].apply(
                    lambda x: 'positive' if len(str(x)) > 50 else 'negative')

            X_vec = vectorizer.fit_transform(df['clean_text'])

            model.fit(X_vec, df['sentiment'])

            df['Sentiment'] = df[review_column].apply(
                lambda x: predict_sentiment(x, model, vectorizer))

            st.subheader("✅ Sentiment Predictions")
            st.write(df[[review_column, 'Sentiment']])

            sentiment_counts = df['Sentiment'].value_counts()
            st.subheader("📈 Sentiment Distribution")
            st.bar_chart(sentiment_counts)

            overall = sentiment_counts.idxmax()
            st.success(f"🔎 Overall Sentiment: {overall}")

            if 'negative' in sentiment_counts:
                st.subheader("⚠ Identified Product Issues")
                negative_reviews = df[df['Sentiment'] == 'negative'][review_column]
                common_problems = extract_problems(negative_reviews)

                if common_problems:
                    st.write("Most common problems identified from negative reviews:")
                    for problem, count in common_problems:
                        st.write(f"- '{problem}': mentioned {count} times")
                else:
                    st.write("No specific problems identified in negative reviews.")

            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "sentiment_results.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
