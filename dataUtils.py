import numpy as np
from typing import Tuple, Any
from numpy.typing import NDArray
from dateutil import parser

import pandas as pd
from sklearn.model_selection import train_test_split

countries = [
    "Taiwan",
    "Guatemala",
    "Colombia",
    "Honduras",
    "Thailand",
    "Ethiopia",
    "Brazil",
    "Costa Rica",
    "Nicaragua",
    "El Salvador",
]

feature_names = countries[:8] + [
    "Altitude",
    "Washed",
    "Pulped natural",
    "Natural",
    "Moisture Percentage",
    "Category One Defects",
    "Category Two Defects",
]
label_names = ["below average", "above average"]


def load_dataset(
    path: str,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[np.int64], NDArray[np.int64]]:
    # some of the preprocessing steps are taken from https://www.kaggle.com/code/tumpanjawat/coffee-eda-geo-cluster-regression#2-|-Exploratory-Data-Analysis-

    df = pd.read_csv("df_arabica_clean.csv")

    # create boolean values for defect classes
    df["Category Two Defects"] = df["Category Two Defects"] > 0
    df["Category One Defects"] = df["Category One Defects"] > 0

    # create one value for altitude
    df = df.join(
        df["Altitude"]
        .str.extract(r"^(?P<min_alt>\d*)\W*[-A~]\W*(?P<max_alt>\d*)$")
        .astype(float)
    )
    df.dtypes
    df["Altitude"] = (
        df[["min_alt", "max_alt"]]
        .mean(axis=1, skipna=True)
        .fillna(df["Altitude"])
        .astype(float)
    )
    df = df.dropna(subset=["Altitude"])

    # Extract the prior year from the "Harvest Year" column
    df["Harvest Year"] = df["Harvest Year"].str.split("/").str[0].str.strip()

    # Convert "Harvest Year" and "Expiration" columns to datetime objects using dateutil parser
    df["Harvest Year"] = pd.to_datetime(df["Harvest Year"], format="%Y")
    df["Expiration"] = df["Expiration"].apply(parser.parse)

    # Mapping the Education
    processing_mapping = {
        "Double Anaerobic Washed": "Washed",
        "Semi Washed": "Washed",
        "Honey,Mossto": "Pulped natural",
        "Double Carbonic Maceration / Natural": "Natural",
        "Wet Hulling": "Washed",
        "Anaerobico 1000h": "Washed",
        "SEMI-LAVADO": "Natural",
        "Natural / Dry": "Natural",
        "Pulped natural / honey": "Pulped natural",
    }
    # Fixing the values in the column
    df["Processing Method"] = df["Processing Method"].map(processing_mapping)
    df["Processing Method"].fillna("Washed", inplace=True)

    df = df.join(pd.get_dummies(df["Processing Method"]))

    # one hot encoded countries
    df = df.join(pd.get_dummies(df["Country of Origin"]))

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=2023)

    X_train = np.array(df_train[feature_names], dtype=np.float64)
    y_train = np.array(df_train["Overall"] > df_train["Overall"].mean(), dtype=np.int64)

    X_test = np.array(df_test[feature_names], dtype=np.float64)
    y_test = np.array(df_test["Overall"] > df_train["Overall"].mean(), dtype=np.int64)
    print(
        f'Mean of overall sensory evaluation of training dataset {df_train["Overall"].mean():.2f}'
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset("df_arabica_clean.csv")
    print(X_train)