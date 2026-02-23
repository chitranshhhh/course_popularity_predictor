import pandas as pd

def load_and_sort_data(filepath):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)
    return df

def filter_advanced_learners(df):
    df_adv = df[df["Skill_Level"] == "Advanced"].copy().reset_index(drop=True)
    return df_adv
