import pandas as pd
from geopy.geocoders import Nominatim
import pickle
import math
geolocator = Nominatim(user_agent="SAD-plantilla")

file_name = "cache.sav"
with open(file_name, "rb") as f:
    cache = pickle.load(f)
global counter
counter = 0
def get_coords(x):
    global counter
    counter += 1
    try:
        if x in cache.keys():
            ret = cache[x]
        else:
            place = geolocator.geocode(x, timeout=10)
            try:
                ret = f"[{place.latitude}, {place.longitude}]"
                print(ret)
            except Exception:
                ret = "Nan"

        cache[x] = ret
        return ret
    except Exception as e:
        print(e)
        print("Failed, saving cache")
        with open(file_name, "wb") as f:
            pickle.dump(cache, f)
        print(f"saved to {file_name}")
        print(counter)
        exit(1)

ml_dataset = pd.read_csv("TweetsTrainDev.csv")

ml_dataset["tweet_location"] = ml_dataset["tweet_location"].apply(lambda x: get_coords(x))
print("Success")
with open(file_name, "wb") as f:
    pickle.dump(cache, f)
print(f"saved to {file_name}")
tweet_coord = list(ml_dataset["tweet_coord"])
tweet_location = list(ml_dataset["tweet_location"])

for (index, coord) in enumerate(tweet_coord):
    print(coord)
    if isinstance(coord, str):
        if coord[0] != "[":
            print(f"replaced {tweet_coord[index]} with {tweet_location[index]}")
            tweet_coord[index] = tweet_location[index]
    elif isinstance(coord, float):
        if math.isnan(coord):
            print(f"replaced {tweet_coord[index]} with {tweet_location[index]}")
            tweet_coord[index] = tweet_location[index]
    else:
        print(type(coord))
        exit(1)

ml_dataset["tweet_coord"] = tweet_coord
del ml_dataset["tweet_location"]

ml_dataset.to_csv("tweets_coord.csv")
