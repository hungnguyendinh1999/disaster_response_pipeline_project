# Disaster Response Pipeline Project


## Content
- [Project Description](#project-description)
- [Run Instructions](#run-instructions)
- [Environment Setup](#environment-setup)


## Project Description
- Data Processing Assessing and cleaning the data, so that it can be utilized by machine learning algorithms. See details in the ETL Notebook.

- Model training Data was passed through a pipeline and a prediction model is made. See details in the ML Notebook.

- Prediction and Visualization Making a web app for prediction and visualization, where user may try some emergency messages and see visualization of distribution of genres and categories.


## Run Instructions:
0. Make sure your environment is setup correctly ([I use Python 3.6 with conda full setup](#environment-setup)).
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Environment Setup
**NOTE: I use Python 3.6. All versions of packages should be in the [requirements.txt](requirements.txt) file. FYI macOS**

I exported the `requirements.txt` file. You can just do `pip install -r requirements.txt` to install required packages for your environment. 

For [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) users:
1. Create Environment:
    - Run this line in Terminal to install all ~290 packages from Anaconda
    `conda create -n env_full anaconda python=3.6`
    
    - Otherwise, run this to create an enviroment and install only the necessary on for this project.
    `conda create -n env --file requirements.txt`
2. Activate Environment
    - `conda activate env_full`
3. When not used, deactivate Environment
    - `conda deactivate`
4. To check all packages installed
    - `conda list -n env_full`