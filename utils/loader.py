import pandas as pd

def load_nurse_profiles(path='data/nurse_profiles.xlsx') -> pd.DataFrame:
    df = pd.read_excel(path)
    df['Name'] = df['Name'].str.strip().str.upper()
    return df


def load_shift_preferences(path='data/nurse_preferences.xlsx') -> pd.DataFrame:
    df = pd.read_excel(path)
    df.rename(columns={df.columns[0]: 'Name'}, inplace=True)
    df.set_index('Name', inplace=True)
    # parse date columns
    cleaned = []
    for col in df.columns:
        # assume format contains YYYY-MM-DD
        dt = pd.to_datetime(str(col).strip().split()[-1], format="%Y-%m-%d").date()
        cleaned.append(dt)
    df.columns = cleaned
    df.index = df.index.str.strip().str.upper()
    return df
