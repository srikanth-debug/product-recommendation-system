def pre_process(df):
    event_weight = {
        "view": 1,
        "click":2,
        "purchase":3
    }
    df['rating'] = df['event'].map(event_weight)
    return df[["user_id","product_id","rating"]]