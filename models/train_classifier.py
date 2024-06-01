import sys
import pandas as pd
from sqlite3 import connect
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import pickle

nltk.download('punkt')
nltk.download('wordnet')



def load_data(database_filepath):
    """
    Loads data from db
    
    Args:
        database_filepath (str): The place where db information is stored.
        
    Returns:
        X,y It is what is used for the Machine Learning part.
    """
    conn = connect(database_filepath)
    df = pd.read_sql('SELECT * FROM MessagesCategories', conn)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    # Ensure all target values are numeric
    y = y.apply(pd.to_numeric, errors='coerce')
    y = y.fillna(0).astype(int)
    
    return X, y

def tokenize(text):
    """
    Tokenizes text that is passed in.
    
    Args:
        text (str): The text you want to tokenize.
        
    Returns:
        list: A list of tokens after processing.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [PorterStemmer().stem(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return words

def build_model():
    """
    Builds the model    
        
    Returns:
    A pipeline used for machine learning
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the model    
    Args:
        model (Model): The model used for machine learning
        X_test: 
        Y_test
        
    Returns:
       The model's score
    """
    y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test.columns):
        print(f'Category: {col}\n', classification_report(Y_test.iloc[:, i], y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Saves the model into a pickle file
    
    Args:
        Model (str): The text you want to tokenize.
        Model path (str): The path where you store the pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main method to run the script    
    Args:
        Model (str): The text you want to tokenize.
        Model path (str): The path where you store the pickle file
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
