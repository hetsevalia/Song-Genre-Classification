# Songe Genre Classfication Using AIML

## Project Overview

This project focuses on classifying song lyrics into five distinct genres: Rock, Jazz, HipHop, Metal, and Country, using machine learning models. By leveraging natural language processing (NLP) techniques and GPU acceleration, we aim to compare the performance of multiple models such as Logistic Regression, Random Forest, and Support Vector Classifier (SVC). The dataset is preprocessed using `nltk` for text cleaning, tokenization, and lemmatization, while feature extraction is handled through `CountVectorizer` and `TfidfVectorizer`.

## Project Structure
- **Data**: The training and testing datasets contain lyrics with corresponding genres. The data is preprocessed to remove noise and is then vectorized for modeling.
- **Models**: The project evaluates models like Logistic Regression, Random Forest, and cuML SVC on both CPU and GPU (via cuML for faster performance).
- **Evaluation**: Classification reports and confusion matrices are generated to analyze model performance for each genre.

## Requirements

### Python Libraries
- `pandas`
- `nltk`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `cudf` (for GPU acceleration)
- `cuML` (for GPU-based Random Forest and SVC)

## Dataset
The dataset consists of song lyrics and their associated genres. The lyrics undergo preprocessing to remove noise and apply tokenization and lemmatization. The project assumes that you have access to train.csv and test.csv files with columns like Lyrics and Genre.

## Preprocessing
The lyrics are preprocessed by:

1. Removing special characters and numbers.
2. Lowercasing the text.
3. Tokenizing and lemmatizing the text.
4. Removing common stopwords using NLTK.
5. Model Training and Evaluation

## The project trains and evaluates several models:

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Classifier (SVC) using GPU with cuML for faster performance.

## Vectorization Techniques:

1. CountVectorizer: A bag-of-words model limiting features to 50,000 words.
2. TfidfVectorizer: Converts lyrics into a TF-IDF representation.

## Results
The models were evaluated based on accuracy and class-wise performance. SVC with cuML demonstrated the best overall performance, particularly when combined with TfidfVectorizer. Other models like Logistic Regression and Random Forest showed slightly lower performance but were more consistent with certain genres.

SVC (cuML + TfidfVectorizer): 67% accuracy.

SVC (cuML + CountVectorizer): 64% accuracy.

## Conclusion: 
Among the models tested, SVC using cuML with TfidfVectorizer delivered the highest accuracy and most balanced performance across genres, making it the optimal model for this task. Logistic Regression and Random Forest classifiers performed well but struggled with certain genres like Country and Rock.
