import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk

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
    X = df['message'].values
    y = df[df.columns[4:]]
    category_names = df.columns[4:]
    
    return X, y, category_names 

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def replace_urls(text):
    """
    Outputs editted version of the input Python str object `text` 
    replacing all urls in text with str 'urlplaceholder'.
    
    INPUT:
        - text - Python str object - a raw text data
        
    OUTPUT:
        - text - Python str object - An editted version of the input data `text` 
          with all urls in text replacing with str 'urlplaceholder'.
    """
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    return text


def tokenize(text):
    """
      Takes a Python string object and outputs list of processed words 
       of the text.
      INPUT:
        - text - Python str object - A raw text data
      OUTPUT:
        - stem_words - Python list object - list of processed words using the input `text`.
    """
    #function with raw text data    
    text = replace_urls(text)
    # Removes punctuations and covert to lower case 
    #text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    text = word_tokenize(text) 
    # Remove stop words
    text = [w for w in text if w not in stopwords.words("english")]
    # Lemmatize verbs by specifying pos
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    
    return text


def build_model():
    '''
    Function specifies the pipeline and the grid search parameters so as to build a
    classification model
     
    Output:  cv: classification model
    '''
    
    # create pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
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