from sklearn.preprocessing import StandardScaler

def preprocess(df):
    df = df.copy()
    scaler = StandardScaler()
    df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])
    return df