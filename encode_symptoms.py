import pandas as pd

symptoms = pd.read_csv('datasets/symptoms.csv')
disease_dataset = pd.read_csv('datasets/dataset.csv').drop("Disease", axis=1)


def encode_symptom(symptom):
    symptom_list = symptoms["Symptom"]
    if symptom in symptom_list:
        return symptom_list.index(symptom)
    else:
        return -1


def encode_all_symptoms():
    columns = get_unique_list(list(symptoms["Symptom"]))
    df = pd.DataFrame(columns=columns)
    lst_symptoms = []
    for i, row in disease_dataset.iterrows():
        cleaned_row = row.to_list()
        cleaned_row = [x for x in cleaned_row if isinstance(x, str)]
        lst_symptoms.append(cleaned_row)
    dict_symptoms = [{symptom.strip(): 1 for symptom in sublist} for sublist in lst_symptoms]
    for i in dict_symptoms:
        df.loc[len(df)] = i
    df.fillna(value=0, inplace=True)
    df.to_csv('datasets/processed.csv', index=False)


def decode_symptom(encoded_symptom):
    symptom_list = symptoms["Symptom"]
    if encoded_symptom is int and 0 <= encoded_symptom <= len(symptom_list):
        return symptom_list[encoded_symptom]
    else:
        return "Invalid encoding"


def get_unique_list(x):
    lst = []
    for i in x:
        if i not in lst:
            lst.append(i)
    return lst


if __name__ == "__main__":
    encode_all_symptoms()
