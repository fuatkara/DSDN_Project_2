import sys
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore') 


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load messages and categories dataset from csv files and merge them  
    into a single dataframe named df variable and returns the merged dataset 
    Input: messages_filepath, categories_filepath
    Output: merged dataframe containing messages and categories dataset
    
    '''
    
    #read and load messages dataset from csv file to a dataframe
    messages = pd.read_csv(messages_filepath)
    
    #reads and load the caegories dataset from csv file to a dataframe
    categories= pd.read_csv(categories_filepath)
    
    #Merge the messages and categories datasets using the common id
    df = pd.merge(left=messages,right=categories, how='inner',on=['id'])
    
    #display the first five datasets
    df.head()

    return df


def clean_data(df):
    """
     Split the values in the categories column on the ; character so that each value becomes a separate column. 
     Use the first row of categories dataframe to create column names for the categories data.
     Rename columns of categories with the new column names.
    """
    
    # Split the values in the categories column on the ; character
    categories = df["categories"].str.split(pat=";",expand=True)
    
    #categories column names were not readable because they are splitted 
    #data shape is like category_name-value so we get column names from any of the rows in the raw data
    # we will got first row for that purpose 
    row= categories[:1]
    #lambda function to extract name exept lats two since they are the value 
    col_names_lambda= lambda x: x[0][:-2]
    category_colnames= list(row.apply(col_names_lambda))
    #column labels asigned by pandas replaced with col names we created
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #check and change all values not equal to 0 or 1 to 1
    for i in categories.columns:
        categories.loc[(categories[i]!=0) & (categories[i]!=1) ,i] = 1
         
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True) 
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace=True) 
    
    return df


def save_data(df, database_filename):
    '''
    Function to save the cleaned dataframe into a sql database with file name 'clean_data'
    
    Input: df, database_filename
    Output: SQL Database 
    '''
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('clean_data', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()