# Fake News Detection

This is a project implemented on a standardised dataset taken from Kaggle.The goal was to enable readers to identify fake news artices while surfing on the web so that they could be well informed with authentic news and also to curb the spreading of false information.

## Overview

This project implements a fake news detection system using an LSTM-based deep learning model. The system is trained on a dataset containing both real and fake news articles, and can predict whether a given news article is likely to be fake or real.

## Features

- Text preprocessing with NLTK
- LSTM-based deep learning model
- Model evaluation with confusion matrix
- Training history visualization
- Easy-to-use prediction function

## Dataset

The project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle, which contains:
- Fake news articles
- Real news articles
- Balanced dataset for training

## Project Structure

```
FakeNewsDetection/
├── data/                    # Dataset directory
│   ├── Fake.csv            # Fake news dataset
│   └── True.csv            # Real news dataset
├── src/                     # Source code
│   ├── data_preprocessing.py  # Data preprocessing functions
│   ├── model.py            # Model implementation
│   └── utils.py            # Utility functions
├── notebooks/               # Jupyter notebooks
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nityareedy/FakeNewsDetection.git
cd FakeNewsDetection
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
- Go to [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Download the dataset
- Place `Fake.csv` and `True.csv` in the `data/` directory

## Usage

1. Run the model training:
```bash
python src/model.py
```

This will:
- Preprocess the data
- Train the LSTM model
- Generate evaluation metrics
- Save the trained model
- Create visualization plots

2. The model will be saved as `fake_news_detector.h5`

## Model Architecture

The model uses:
- Embedding layer
- Two LSTM layers
- Dropout for regularization
- Dense layer with sigmoid activation

## Results

The model generates:
- Classification report
- Confusion matrix
- Training history plots

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Libraries: TensorFlow, Keras, NLTK, scikit-learn 
