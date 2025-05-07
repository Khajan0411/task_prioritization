# preprocess.py
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from datetime import datetime

def preprocess_data(df):
    le_urgency = LabelEncoder()
    le_category = LabelEncoder()
    le_priority = LabelEncoder()

    df['Urgency'] = le_urgency.fit_transform(df['Urgency'])
    df['Category'] = le_category.fit_transform(df['Category'])
    df['Priority'] = le_priority.fit_transform(df['Priority'])

    df['Deadline'] = pd.to_datetime(df['Deadline'])
    df['DaysUntilDeadline'] = (df['Deadline'] - datetime.today()).dt.days
    df.drop('Deadline', axis=1, inplace=True)

    X = df.drop('Priority', axis=1)
    y = df['Priority']
    
    return X, y, le_priority
