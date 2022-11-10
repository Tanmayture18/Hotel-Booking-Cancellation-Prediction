# Importing necessary libraries
import pandas as pd
import numpy as np
import pickle

if __name__=='__main__':

    # Importing dataset
    df=pd.read_csv('hotel_bookings.csv')

    # Null values imputation
    df['country']=df['country'].fillna(df['country'].mode()[0])
    df['company']=df['company'].fillna(df['company'].mean())
    df['agent']=df['agent'].fillna(df['agent'].mean())
    df['children']=df['children'].fillna(df['children'].mean())

    # Dropping unnecessary columns
    df=df[['is_canceled','lead_time','adults','children','is_repeated_guest','previous_cancellations','meal','deposit_type','required_car_parking_spaces','customer_type']]

    # Label Encoding
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    df['customer_type']=le.fit_transform(df['customer_type'])
    df['meal']=le.fit_transform(df['meal'])
    df['deposit_type']=le.fit_transform(df['deposit_type'])

    # Separating dependent and independent variables
    X=df.drop('is_canceled',axis=1)
    y=df.is_canceled

    # Performing train-test split
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    

    # Fitting algorithm
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier()
    clf.fit(X_train,y_train)

    # Pickling
    file=open('model.pk1','wb')
    pickle.dump(clf,file)   
    file.close() 



