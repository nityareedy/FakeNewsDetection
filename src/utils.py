import numpy as np
import matplotlib.pyplot as plt
import re
from keras.preprocessing.sequence import pad_sequences

def plot_training_history(history):
    """
    Plot the training and validation accuracy/loss
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def predict_news(model, tokenizer, text, max_length=500):
    """
    Predict whether a given news article is fake or real
    """
    # Preprocess the text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    
    return {
        'prediction': 'Fake' if prediction < 0.5 else 'Real',
        'confidence': prediction if prediction >= 0.5 else 1 - prediction
    } 