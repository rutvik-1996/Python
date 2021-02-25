"""
ALY6140 71697 Analytics Systems Technology Fall A
College of Professional Studies, Northeastern University, Boston, MA 02115.
WEEK 6 : FINAL PROJECT
Submission date : 10/24/20
Copyright (c) 2020
Licensed
Written by Aishwarya Subramanya, Rutvi Kantawala, Nikhil Sakinal,Spurthi Patnam
Instructor name : Kamen Madjarov

"""
#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('https://files.consumerfinance.gov/ccdb/complaints.csv.zip', sep=',') #reading the csv file from the url
    df["Date received"] = pd.to_datetime(df["Date received"])
    df['Date sent to company'] = pd.to_datetime(df["Date sent to company"])
    df['Year'] = df['Date received'].dt.year
    df['Month'] = df['Date received'].dt.month
    df['Week Number']=df['Date received'].dt.week
    df['State'] = df['State'].fillna("Not given")
    df['Product'] = df['Product'].fillna("Not given")
    df['Sub-product'] = df['Sub-product'].fillna("Not given")
    df['Issue'] = df['Issue'].fillna("Not given")
    df['Sub-issue'] = df['Sub-issue'].fillna("Not given")
    df['Consumer complaint narrative'] = df['Consumer complaint narrative'].fillna("Not given")
    df['Company public response'] = df['Company public response'].fillna("Not given")
    df['Company response to consumer'] = df['Company response to consumer'].fillna("Not given")
    df['Consumer consent provided?'] = df['Consumer consent provided?'].fillna("Not given")
    df['Consumer disputed?'] = df['Consumer disputed?'].fillna("Not given")
    df['Tags'] = df['Tags'].fillna("Not given")
    complaints_data = df[(df['Date received'] > '2018-1-1') & (df['Date received'] <= '2020-10-31')] #Subsetting the data from the original dataframe and considering last 3 years data for our analysis
    complaints_data['category_id'] = complaints_data['Product'].factorize()[0]

#Function to map month numbers to the actual month name

    month_map = {1: 'January', 2: 'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August',
             9:'September', 10:'October', 11:'November', 12:'December'}
    def mapper(Month):
        return month_map[Month]

    complaints_data['Month']=complaints_data['Month'].apply(mapper)


    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
    from sklearn.model_selection import train_test_split
    from sklearn import model_selection, preprocessing, linear_model, metrics
    from sklearn.linear_model import LogisticRegression

    x_train, x_test, y_train, y_test = model_selection.train_test_split(complaints_data['Consumer complaint narrative'], complaints_data['Product'])

    encoder = preprocessing.LabelEncoder() #Label encoding
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    tfidf_vec = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vec.fit(complaints_data['Consumer complaint narrative'])
    xtrain_tfidf = tfidf_vec.transform(x_train)
    xtest_tfidf = tfidf_vec.transform(x_test)


    #Model Building 
    model = linear_model.LogisticRegression(solver='newton-cg', multi_class='multinomial').fit(xtrain_tfidf,  y_train)

    # Predict values using the test data
    accuracy = metrics.accuracy_score(model.predict(xtest_tfidf), y_test)
    print("Accuracy: ", accuracy)
    # View the accuracy of the model against the test data.

    print(metrics.classification_report(y_test, model.predict(xtest_tfidf), target_names=complaints_data['Product'].unique()))
    print()


if __name__ == '__main__':
    main()






