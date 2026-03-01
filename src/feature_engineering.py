import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_features(df):

    # Convert dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

    # Date features
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Quarter'] = df['Order Date'].dt.quarter
    df['Order Day'] = df['Order Date'].dt.day
    df['Day of Week'] = df['Order Date'].dt.dayofweek

    # Shipping time
    df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Encode categorical variables
    le = LabelEncoder()

    df['Category_encoded'] = le.fit_transform(df['Category'])
    df['Region_encoded'] = le.fit_transform(df['Region'])
    df['Sub_Category_encoded'] = le.fit_transform(df['Sub-Category'])
    df['Segment_encoded'] = le.fit_transform(df['Segment'])

    return df