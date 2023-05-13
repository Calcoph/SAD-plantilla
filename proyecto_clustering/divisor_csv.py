import pandas as pd
import datetime

CSV_TIME_FORMAT = "%Y-%m-%d %H:%M:%S %z"

def mantain_only(x, value):
    if x == value:
        return value
    else:
        return ""

if __name__ == "__main__":
    if False:
        ml_dataset = pd.read_csv("TweetsTrainDev.csv")
        ml_dataset.drop(ml_dataset[ml_dataset["airline"] != "United"].index, inplace = True)
        ml_dataset.drop(ml_dataset[ml_dataset["airline_sentiment"] != "negative"].index, inplace = True)

        ml_dataset.to_csv("TweetsTrainDev_united_neg.csv", index=False)
    elif False:
        ml_dataset = pd.read_csv("TweetsTrainDev.csv")
        ml_dataset.drop(ml_dataset[ml_dataset["airline"] != "Southwest"].index, inplace = True)
        ml_dataset.drop(ml_dataset[ml_dataset["airline_sentiment"] != "negative"].index, inplace = True)

        ml_dataset.to_csv("TweetsTrainDev_southwest_neg.csv", index=False)
    elif False:
        ml_dataset = pd.read_csv("TweetsTrainDev.csv")
        ml_dataset.drop(ml_dataset[ml_dataset["airline"] != "Southwest"].index, inplace = True)
        ml_dataset["airline_sentiment_pos"] = ml_dataset["airline_sentiment"].copy()
        ml_dataset["airline_sentiment_neg"] = ml_dataset["airline_sentiment"].copy()
        ml_dataset["airline_sentiment_neut"] = ml_dataset["airline_sentiment"].copy()
        ml_dataset = ml_dataset[["tweet_created","airline_sentiment_pos","airline_sentiment_neg","airline_sentiment_neut"]]
        ml_dataset["airline_sentiment_pos"] = ml_dataset["airline_sentiment_pos"].apply(lambda x: mantain_only(x, "positive"))
        ml_dataset["airline_sentiment_neg"] = ml_dataset["airline_sentiment_neg"].apply(lambda x: mantain_only(x, "negative"))
        ml_dataset["airline_sentiment_neut"] = ml_dataset["airline_sentiment_neut"].apply(lambda x: mantain_only(x, "neutral"))
        ml_dataset["tweet_created"] = ml_dataset["tweet_created"].apply(lambda x: datetime.datetime.strptime(x, CSV_TIME_FORMAT).strftime("%Y/%m/%d"))

        ml_dataset.to_csv("TweetsTrainDev_southwest_fecha_reviews.csv", index=False)
    elif True:
        ml_dataset = pd.read_csv("TweetsTrainDev.csv")
        ml_dataset.drop(ml_dataset[ml_dataset["airline"] != "Southwest"].index, inplace = True)
        ml_dataset = ml_dataset[["tweet_created","airline_sentiment"]]
        ml_dataset["tweet_created"] = ml_dataset["tweet_created"].apply(lambda x: datetime.datetime.strptime(x, CSV_TIME_FORMAT).strftime("%Y/%m/%d"))

        ml_dataset.to_csv("TweetsTrainDev_southwest_fecha_reviews.csv", index=False)