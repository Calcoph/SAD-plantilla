import pandas as pd

ml_dataset = pd.read_csv("tweets_coord.csv")
coords = list(ml_dataset["tweet_coord"])

latitudes = []
longitudes = []
for coord in coords:
    try:
        separated_coord = coord.split(",")
        if len(separated_coord) == 2:
            latitude = separated_coord[0][1:]
            longitude = separated_coord[1][:-1]
            latitudes.append(latitude)
            longitudes.append(longitude)
        else:
            latitudes.append(coord)
            longitudes.append(coord)
    except AttributeError:
        latitudes.append(coord)
        longitudes.append(coord)

ml_dataset["tweet_lat"] = latitudes
ml_dataset["tweet_long"] = longitudes
del ml_dataset["tweet_coord"]

ml_dataset.to_csv("tweets_coord_separated.csv")
