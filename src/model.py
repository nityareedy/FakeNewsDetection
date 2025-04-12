import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_preprocess_data

def create_model(vocab_size, max_length):
    """
    Create and compile the LSTM model
    """
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_length),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, X_test, y_train, y_test):
    """
    Train the model and return the trained model and history
    """
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    max_length = 500
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    
    # Create and train the model
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(vocab_size, max_length)
    
    history = model.fit(
        X_train_pad,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test_pad, y_test),
        verbose=1
    )
    
    return model, history, tokenizer

def evaluate_model(model, X_test_pad, y_test):
    """
    Evaluate the model and plot the results
    """
    # Make predictions
    y_pred = (model.predict(X_test_pad) > 0.5).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train the model
    model, history, tokenizer = train_model(X_train, X_test, y_train, y_test)
    
    # Evaluate the model
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=500, padding='post')
    evaluate_model(model, X_test_pad, y_test)
    
    # Save the model
    model.save('fake_news_detector.h5')
    print("\nModel saved as 'fake_news_detector.h5'") 