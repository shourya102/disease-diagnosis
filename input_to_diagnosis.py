import pandas as pd
from joblib import load

from encode_symptoms import get_unique_list


def input_to_diagnosis(symptoms):
    model = load('models/dt.joblib')
    columns = get_unique_list(list(pd.read_csv("datasets/symptoms.csv")["Symptom"]))
    test_df = pd.DataFrame(columns=columns)
    test_df.loc[0] = symptoms
    test_df.fillna(0, inplace=True)
    predictions = model.predict(test_df)
    le = load('le.joblib')
    return le.inverse_transform(predictions)[0]
