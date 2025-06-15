# Detecting-Spam-Emails-Using-Tensorflow
📧 Spam Email Detection using TensorFlow
This project demonstrates how to classify emails as spam or ham (not spam) using a deep learning model built with TensorFlow and Keras. The workflow includes data preprocessing, model building, training, evaluation, and visualization — all implemented in a single Jupyter Notebook.

## 📁 Dataset
The project uses a dataset containing 5,171 emails, each labeled as "spam" or "ham".

Columns: text (email body), label (spam/ham)

The dataset is imbalanced (more ham than spam), so it is downsampled for balance.

## 🧰 Requirements
Install dependencies using:

pip install tensorflow pandas numpy matplotlib seaborn nltk wordcloud scikit-learn
NLTK resources used:

python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('punkt')


## 📊 Project Workflow
1. Data Preprocessing
Load dataset and balance classes

Clean text: lowercasing, removing punctuation, stopword removal

Tokenize and pad email sequences using Keras

2. Model Building
Embedding Layer

LSTM Layer

Dense Output Layer with Sigmoid activation

3. Model Compilation and Training
Loss: binary_crossentropy

Optimizer: adam

Metrics: accuracy

Includes EarlyStopping and ReduceLROnPlateau callbacks

4. Evaluation
Final accuracy and loss on test set

Accuracy ~97%

Visualizations: training/validation loss and accuracy

## 📈 Visualizations
WordClouds for spam and ham emails

Label distribution bar chart

Training history plots (accuracy & loss)

## 🧪 Potential Improvements
Use Bidirectional LSTM or GRU

Add TF-IDF or pre-trained embeddings (GloVe, Word2Vec)

Implement advanced balancing (SMOTE, oversampling)

Evaluate with precision, recall, F1-score

Save and deploy model with Flask or FastAPI

