import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse # Added for CLI

# Ensure NLTK resources are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    nltk.download('wordnet')

def load_and_inspect_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame, prints some basic information,
    and returns the DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        print("DataFrame shape:", df.shape)
        print("\nFirst 5 rows of the DataFrame:")
        print(df.head())
        print("\nValue counts of the 'label' column:")
        print(df['label'].value_counts())
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def preprocess_text(text_series):
    """
    Processes a pandas Series of text messages.

    Args:
        text_series (pd.Series): A pandas Series containing text messages.

    Returns:
        pd.Series: A new pandas Series with processed messages.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    processed_messages = []
    for message in text_series:
        message = message.lower()
        message = re.sub(r'[^\w\s]', '', message)
        tokens = message.split()
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        processed_messages.append(" ".join(processed_tokens))
        
    return pd.Series(processed_messages)

def train_model(X_features, y_labels, test_size=0.2, random_state=42):
    """
    Splits data, trains a Multinomial Naive Bayes model.

    Args:
        X_features: Feature set (e.g., TF-IDF features).
        y_labels: Target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: Trained model, X_test features, y_train labels, y_test labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, X_test, y_train, y_test

def evaluate_model(model, X_test_features, y_test_true):
    """
    Evaluates the trained model and prints performance metrics.

    Args:
        model: The trained classification model.
        X_test_features: The TF-IDF features of the test set.
        y_test_true: The true labels for the test set.
    """
    y_pred = model.predict(X_test_features)

    accuracy = accuracy_score(y_test_true, y_pred)
    precision = precision_score(y_test_true, y_pred, zero_division=0)
    recall = recall_score(y_test_true, y_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_pred, zero_division=0)

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    cm = confusion_matrix(y_test_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham (0)', 'Spam (1)'], yticklabels=['Ham (0)', 'Spam (1)'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png') 
    print("\nConfusion matrix saved as confusion_matrix.png")
    # plt.show() # Commented out for non-GUI environments

def predict_message(raw_text, vectorizer, model, preprocessor):
    """
    Predicts if a raw text message is Ham or Spam.

    Args:
        raw_text (str): The input message.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        model (MultinomialNB): The trained Naive Bayes model.
        preprocessor (function): The preprocess_text function.

    Returns:
        str: "Ham" or "Spam".
    """
    processed_text_series = preprocessor(pd.Series([raw_text]))
    processed_message = processed_text_series.iloc[0]
    message_tfidf = vectorizer.transform([processed_message])
    prediction = model.predict(message_tfidf)
    return "Ham" if prediction[0] == 0 else "Spam"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and use a spam classifier.")
    parser.add_argument('--dataset', type=str, default='spam_dataset.csv',
                        help='Path to the dataset CSV file (default: spam_dataset.csv)')
    args = parser.parse_args()
    dataset_path = args.dataset

    # Initialize model and vectorizer so they are in scope
    model = None
    tfidf_vectorizer = None
    # X_test_features and y_test_labels are primarily for evaluation, not direct CLI use beyond that.
    # They are populated during the training/evaluation phase.

    try:
        data_df = load_and_inspect_data(dataset_path)
        if data_df is not None:
            if 'text' in data_df.columns:
                data_df['processed_text'] = preprocess_text(data_df['text'])
                print("\nFirst 5 processed messages:")
                print(data_df['processed_text'].head())

                # Initialize and fit TfidfVectorizer here
                tfidf_vectorizer = TfidfVectorizer(max_features=3000)
                X_tfidf = tfidf_vectorizer.fit_transform(data_df['processed_text'])

                if 'label' in data_df.columns:
                    y_mapped_labels = data_df['label'].map({'ham': 0, 'spam': 1})
                    if y_mapped_labels.isnull().any():
                        print("Error: 'label' column contains values other than 'ham' or 'spam', or has missing values after mapping. Please check your dataset.")
                    else:
                        # Assign trained model to the 'model' variable
                        trained_model, X_test_features, _, y_test_labels = train_model(X_tfidf, y_mapped_labels)
                        model = trained_model # Assign to the model variable in the broader scope
                        print("\nModel trained successfully!")
                        
                        evaluate_model(model, X_test_features, y_test_labels)
                else:
                    print("Error: 'label' column not found in the dataset.")
            else:
                print("Error: 'text' column not found in the DataFrame.")
        else:
            print(f"DataFrame loading failed from '{dataset_path}'. Please provide a valid path to your dataset.")
    except FileNotFoundError:
        # This specific FileNotFoundError for dataset_path is already handled by load_and_inspect_data,
        # but keeping a general catch here for robustness.
        print(f"Error: The dataset file '{dataset_path}' was not found. Please ensure the file exists or provide a valid path.")
    except Exception as e:
        print(f"An unexpected error occurred during the data loading/training phase: {e}")

    # Prediction loop
    if model is not None and tfidf_vectorizer is not None:
        print("\nStarting interactive prediction mode...")
        while True:
            user_message = input("\nEnter a message to classify (or type 'exit' to quit): ")
            if user_message.lower() == 'exit':
                break
            try:
                prediction = predict_message(user_message, tfidf_vectorizer, model, preprocess_text)
                print(f"Prediction: {prediction}")
            except Exception as e:
                print(f"An error occurred during prediction: {e}")
                print("Please ensure the model and vectorizer are correctly loaded.")
    else:
        print("\nExiting: Model not trained or vectorizer not initialized due to errors in the data loading or training phase.")
