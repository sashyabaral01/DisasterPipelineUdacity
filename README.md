# DisasterPipelineUdacity
# DisasterPipelineUdacity



# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage



Visualization 

![Screenshot 2024-06-01 at 15-16-59 Disasters](https://github.com/sashyabaral01/DisasterPipelineUdacity/assets/37986335/f56b70e9-7f7b-4d73-a870-4689571fee26)


This is the charts I created. It shows the percentage of distrubtion of messsages in the form of a pie chart and a barchart. It also counts the occurances of the different types of message requests.

![Screenshot 2024-06-01 at 15-18-52 Disasters](https://github.com/sashyabaral01/DisasterPipelineUdacity/assets/37986335/b2746bcc-8705-4549-a5cc-18f553423f55)


This shows the classification of the message after typing in the follwing messages: we are starving and we need to eat. This shows the classification of messages it belongs to. 
