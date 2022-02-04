import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
import joblib 
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet','stopwords'])

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import PorterStemmer

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


import warnings
warnings.filterwarnings('ignore') 

def load_data(database_filepath):
    
    """
    Function to Load dataset from database sql database (database_filepath) and split the dataframe into X and y variable
    Input: Database filepath
    Output: Returns the Features and target variables X and Y along with target columns names catgeory_names
    """
    
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql('clean_data',con=engine)
    # allocate the feature and target variables to X and y
    X = df['message']
    y = df[df.columns[5:]]
    
    
    return X, y, category_names 

def tokenize(text):
    # normalize text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    
    # Reduce words to their stems
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in stemmed]
    
    return lemmed


def build_model():
    '''
    Function specifies the pipeline and the grid search parameters so as to build a
    classification model
     
    Output:  cv: classification model
    '''
    
    # create pipeline
    ds_pipe = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))),
    ])
    
  
    # my computer too considerable amount of time but never finished building it. 
    # In oder to finish the project, I skiped this part.
    #parameters = {
        #'clf__estimator__criterion':['gini','entropy'],  
        #'clf__estimator__min_samples_split':[10,110],
        #'clf__estimator__max_depth':[None,100,500]
             #}

    # choose parameters
    parameters = {
        'clf__estimator__n_estimators': [100, 250]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
      output: prints classification report 
    """
    
    y_pred = model.predict(X_test)
    
    report = classification_report(Y_test, y_pred,target_names = category_names)
        
    print(report)
    return report


def save_model(model, model_filepath):
    
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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