import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess the text by:
    1. Converting to lowercase
    2. Removing special characters
    3. Tokenizing
    4. Removing stopwords
    5. Lemmatizing
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def load_and_preprocess_data():
    """
    Load and preprocess the dataset
    """
    # Load the dataset
    fake_news = pd.read_csv('data/Fake.csv')
    real_news = pd.read_csv('data/True.csv')
    
    # Add labels
    fake_news['label'] = 0  # 0 for fake news
    real_news['label'] = 1  # 1 for real news
    
    # Combine datasets
    df = pd.concat([fake_news, real_news], axis=0)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['label'],
        test_size=0.2,
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Data preprocessing completed successfully!")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}") 