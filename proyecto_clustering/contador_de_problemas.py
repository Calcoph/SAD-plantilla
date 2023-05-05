import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_numeric, STOPWORDS
import nltk
nltk.download("wordnet")

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases

import crear_modelo_config

if __name__ == "__main__":
    ml_dataset = pd.read_csv("TweetsTrainDev.csv")
    ml_dataset.drop(ml_dataset[ml_dataset["airline"] != "United"].index, inplace = True)
    ml_dataset.drop(ml_dataset[ml_dataset["airline_sentiment"] != "negative"].index, inplace = True)

    razones = list(ml_dataset["negativereason"])
    diccionario_razones = {

    }

    for razon in razones:
        if razon in diccionario_razones:
            diccionario_razones[razon] += 1
        else:
            diccionario_razones[razon] = 1
    
    lista_razones = list(diccionario_razones.items())
    lista_razones.sort(key= lambda x: x[1], reverse=True)
    for (key, value) in lista_razones:
        lista_razones.append
        print(f"{key}: {value}")
