import streamlit as st
import pandas as pd
from mlmorph import Analyser
from collections import defaultdict
import math

# Initialize the Analyser for Malayalam
st.write("Initializing application...")
try:
    analyser = Analyser()
    st.write("Malayalam analyser initialized.")
except Exception as e:
    st.error(f"Error initializing Malayalam analyser: {e}")

# Tokenization Function
def tokenize_malayalam(text):
    """
    Tokenize Malayalam text using mlmorph.
    """
    tokens = []
    for word in str(text).split():
        analyses = analyser.analyse(word)
        if analyses:
            root_form = analyses[0][0]  # Extract root form
            tokens.append(root_form)
        else:
            tokens.append(word)  # Fallback to original word
    return tokens

# Training the Naive Bayes Model
def train_naive_bayes_unigram(df):
    """
    Train a Naive Bayes classifier using tokenized unigrams.
    """
    class_counts = defaultdict(int)
    class_word_counts = defaultdict(lambda: defaultdict(int))
    vocabulary = set()

    for _, row in df.iterrows():
        tokens = row["tokenized_headings"]  # Use tokenized text
        label = row["label"]

        class_counts[label] += 1
        for token in tokens:
            class_word_counts[label][token] += 1
            vocabulary.add(token)

    return class_counts, class_word_counts, vocabulary

# Calculate Probabilities with Laplace Smoothing
def calculate_class_probabilities_unigram(class_counts, class_word_counts, vocabulary, alpha=1.0):
    """
    Calculate class and unigram probabilities with Laplace smoothing.
    """
    total_words_in_class = {label: sum(class_word_counts[label].values()) for label in class_counts}
    class_probabilities = {}

    for label in class_counts:
        word_probs = {}
        for token in vocabulary:
            token_count = class_word_counts[label].get(token, 0)
            word_probs[token] = (token_count + alpha) / (total_words_in_class[label] + alpha * len(vocabulary))

        class_prob = class_counts[label] / sum(class_counts.values())
        class_probabilities[label] = (class_prob, word_probs)

    return class_probabilities

# Prediction Function
def predict_unigram(text, class_probabilities, vocabulary):
    """
    Predict the label for a given text using the trained unigram model.
    """
    tokens = tokenize_malayalam(text)  # Tokenize the input text
    label_scores = {}

    for label, (class_prob, word_probs) in class_probabilities.items():
        log_prob = math.log(class_prob)  # Start with the log of the class probability
        for token in tokens:
            if token in vocabulary:
                log_prob += math.log(word_probs.get(token, 1 / (sum(word_probs.values()) + len(vocabulary))))
        label_scores[label] = log_prob

    return max(label_scores, key=label_scores.get)

# Streamlit App
st.title("Malayalam Text Classification")
st.write("This app predicts the category of a Malayalam text using unigram tokenization.")

# Load the Preloaded Dataset
@st.cache_data
def load_dataset():
    try:
        # Load the preloaded train.csv file
        df = pd.read_csv("D:\\cl\\train.csv")
        st.write("Dataset loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_dataset()

if df is not None:
    # Preprocess the Dataset
    try:
        df['tokenized_headings'] = df['headings'].apply(tokenize_malayalam)
        st.write("Dataset tokenization complete. Here's a sample of the tokenized dataset:")
        st.write(df[['headings', 'tokenized_headings']].head())
    except Exception as e:
        st.error(f"Error during tokenization: {e}")

    # Train the Naive Bayes Model
    try:
        class_counts, class_word_counts, vocabulary = train_naive_bayes_unigram(df)
        st.write("Naive Bayes model trained successfully.")
        st.write(f"Vocabulary size: {len(vocabulary)}")
    except Exception as e:
        st.error(f"Error during model training: {e}")

    # Calculate Class Probabilities
    try:
        class_probabilities = calculate_class_probabilities_unigram(class_counts, class_word_counts, vocabulary)
        st.write("Class probabilities calculated.")
    except Exception as e:
        st.error(f"Error during probability calculation: {e}")

    # User Input Section
    st.write("Enter text to classify:")
    input_text = st.text_area("Enter Malayalam Text", height=150)

    if st.button("Classify"):
        st.write(f"Input Text: {input_text}")
        if input_text.strip():
            try:
                predicted_label = predict_unigram(input_text, class_probabilities, vocabulary)
                st.write(f"**Predicted Category:** {predicted_label}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.write("Please enter text to classify.")
else:
    st.write("Dataset not loaded. Please check the train.csv file.")
