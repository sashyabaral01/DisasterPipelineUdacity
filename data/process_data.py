import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads data from csv
    
    Args:
        The csv files to read        
    Returns:
        A dataframe.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id") 
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]

    category_columns = row.str.split('-', expand=True)[0]
    categories.columns = category_columns
    for column in categories:
        #print("Current column: " + column)
    #categories[column] = categories[column].str.split('-',expand=True)
        categories[column] = categories[column].apply(lambda x: x.split('-')[-1])
    # convert column from string to numeric
        categories[column] = pd.to_numeric (categories[column], errors = 'coerce') 
        df = pd.concat([df, categories], axis=1)   
    print(df['genre'].value_counts())

    return df


    







def clean_data(df):

    """
    Cleans the data    
    Args:
        dataframe        
    Returns:
        Returns the clean dataframe
    """
    df = df.drop_duplicates(subset=['message'])
    return df



def save_data(df, database_filename):


    """
    Loads data from db
    
    Args:
        dataframe
        database_filename: The desired saved db filename
        
    Returns:
        Returns the database after being saved.
    """

    engine = create_engine('sqlite:///'+ str (database_filename))
    df.to_sql('MessagesCategories', engine, index=False, if_exists = 'replace')
    return df

  


def main():
    """
    Main method    
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print("Before cleaning")
        print(df['genre'].value_counts())
        print('Cleaning data...')
        df = clean_data(df)

        print(df.head())
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
        
        print("Final steps")
        print(df['genre'].value_counts())


    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()