import pandas as pd
from google.cloud import storage
from sklearn import linear_model
import numpy as np
import joblib
from TFM_TrainAtScalePipeline.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, STORAGE_LOCATION

### GCP configuration - - - - - - - - - - - - - - - - - - -
#All the variables in params.py


def get_data():
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')


def compute_distance(df,
                     start_lat="pickup_latitude",
                     start_lon="pickup_longitude",
                     end_lat="dropoff_latitude",
                     end_lon="dropoff_longitude"):
    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)), np.radians(
        df[start_lon].astype(float))
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)), np.radians(
        df[end_lon].astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(
        dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def preprocess(df):
    """method that pre-process the data"""
    df["distance"] = compute_distance(df)
    X_train = df[["distance"]]
    y_train = df["fare_amount"]
    return X_train, y_train


def train_model(X_train, y_train):
    """method that trains the model"""
    rgs = linear_model.Lasso(alpha=0.1)
    rgs.fit(X_train, y_train)
    print("trained model")
    return rgs


def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')


def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(
        f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}"
    )


if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data()

    # preprocess data
    X_train, y_train = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    reg = train_model(X_train, y_train)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(reg)
