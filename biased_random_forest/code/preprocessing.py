import pandas as pd
import random


# Split Train and Test Sets According to Specified Test Ratio
def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    # Sample the Test Set Randomly
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def feature_engineering(frame):
    # Credits to Kaggle Kernel Author Vincent Lugat
    # https://www.kaggle.com/vincentlugat/pima-indians-diabetes-eda-prediction-0-906#4.-New-features-(16)-and-EDA

    # Feature Engineering Intelligent Interaction Features
    frame['F1'] = frame['BMI'] * frame['SkinThickness']
    frame['F2'] = frame['Pregnancies'] / frame['Age']
    frame['F3'] = frame['Glucose'] / frame['DiabetesPedigreeFunction']
    frame['F4'] = frame['Age'] * frame['DiabetesPedigreeFunction']
    frame['F5'] = frame['Age'] / frame['Insulin']

    # Feature Engineering Intelligent Binary Indicator Features
    for col in ['F6', 'F7', 'F8', 'F9', 'F10', 'F11']:
        frame.loc[:, col] = 0
    frame.loc[(frame['Age'] <= 30) & (frame['Glucose'] <= 120), 'F6'] = 1
    frame.loc[(frame['BMI'] <= 30), 'F7'] = 1
    frame.loc[(frame['Glucose'] <= 105) & (frame['BloodPressure'] <= 80), 'F8'] = 1
    frame.loc[(frame['BMI'] < 30) & (frame['SkinThickness'] <= 20), 'F9'] = 1
    frame.loc[(frame['Glucose'] <= 105) & (frame['BMI'] <= 30), 'F10'] = 1
    frame.loc[(frame['BloodPressure'] < 80), 'F11'] = 1

    return frame


def preprocess_and_split(filepath, test_size=0.2):
    # Import the Data
    df = pd.read_csv(filepath)
    df['label'] = df.Outcome
    df = df.drop("Outcome", axis=1)

    # Change Spaces to Underscores
    column_names = []
    for column in df.columns:
        name = column.replace(" ", "_")
        column_names.append(name)
    df.columns = column_names

    train_df, test_df = train_test_split(df, test_size)

    # Clean the Data
    # Impute Median of Data on that Variable with that Outcome into Zeroes
    for outcome in [0, 1]:
        for frame in [train_df, test_df]:
            for var in ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness', 'Insulin']:

                median_target = train_df.loc[(train_df[var] != 0) & (train_df['label'] == outcome), var].median()

                frame.loc[(frame[var] == 0) & (frame['label'] == outcome), var] = frame.loc[
                    (frame[var] == 0) & (frame['label'] == outcome), var].replace(to_replace=0, value=median_target)

    # Now Engineer Features Now that Zeros Were Imputed Into
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # Move Label Column to the End Again
    train_df = train_df[[c for c in train_df if c != 'label'] + ['label']]
    test_df = test_df[[c for c in test_df if c != 'label'] + ['label']]

    return train_df, test_df