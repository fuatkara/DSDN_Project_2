# Disaster Response Application
 

# 1- Introduction of the project
when a disaster happened, It's important to filter the most important messages after responsible agencies receiving millions of direct or social media communications. Often, only one in a thousand messages is relevant for disaster response professionals. Typically, in response to disasters, Different organizations take care of different parts of the problem. One organization takes care of water, another takes care of blocked roads, another takes care of medical supplies. 

In this repo / study, Thousands of real messages of disaster data from Figure Eight will be analysed, which contains pre-labeled tweets and text messages from real-life disasters, to create a model for an API that classifies disaster messages.

# 2- Prerequisites
To install the flask app, you need:

python3
python packages in the requirements.txt file
Install the packages with

pip install -r requirements.txt
To create an environment using: conda create --name --file requirements.txt

# 3- Project Components
There are three components for this project:

ETL Pipeline: First I will repair the data with the ETL pipeline that process messages and category data from CSV file and load them into SQLite database. In the Python script, process_data.py, you will find the data cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

ML Pipeline: Use the machine learning pipeline to raed data from the SQLite database to create and save a multi-output supervised learning model. In the Python script, train_classifier.py, you will find the machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

Flask Web App: A web application is created, which use the trained model(the pickle file) to classify incoming messages where an emergency worker can input a new message and get classification results in several categories.


# Instructions
1. Download the files or clone this repository
git clone https://github.com/fuatkara/DSNN_Project_2
2. Execute the scripts

a. Open a terminal

b. Direct to the project's root directory

c. Run the following commands:

To run ETL pipeline that cleans data and stores in database

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseApp.db
To run ML pipeline that trains classifier and saves
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

d. Go to the app's directory and run the command

cd app

python run.py
