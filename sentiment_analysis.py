# https://github.com/savasy/TurkishSentimentAnalysis

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sa= pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)


import pandas as pd
reviews = pd.read_csv("reviews.csv")
len(reviews)
comment_df = list(reviews["comment"])

sentiment_commnet = []
for line in comment_df:
    p = sa(line)
    # print(p)
    bit = p[0]['label']
    if bit == "negative":
        sentiment_commnet.append(0)
    elif bit == "positive":
        sentiment_commnet.append(1)
    else:
        sentiment_commnet.append(2)
    



