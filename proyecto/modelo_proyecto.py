import pandas as pd
import numpy as np
import sys
import json
import jsonschema
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import TypeVar, cast
from sklearn.preprocessing import LabelEncoder
import pickle
#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#print(nltk.find('corpora/wordnet.zip'))
#from nltk import word_tokenize
#from nltk.stem import WordNetLemmatizer
from emosent import get_emoji_sentiment_rank, EMOJI_SENTIMENT_DICT
from sklearn.feature_extraction import text
"""
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
"""

NEGATIVE_EVALUATION = True

class Enum:
    pass

class MaximizeValue(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    FSCORE = "f1-score"
    ACCURACY = "accuracy"

class StopWords:
    def __init__(
            self,
            start_from_english: bool,
            add_words: list[str],
            remove_words: list[str]
        ) -> None:

        self.start_from_english = start_from_english
        self.add_words = add_words
        self.remove_words = remove_words

    def any_change(self) -> bool: 
        ret = False
        if self.start_from_english:
            ret = True
        if len(self.add_words) > 0:
            ret = True
        if len(self.remove_words) > 0:
            ret = True

        return ret

class ImputeMethod(Enum):
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    CREATE_CATEGORY = "CREATE_CATEGORY"
    MODE = "MODE"
    CONSTANT = "CONSTANT"

class ImputeFeature:
    def __init__(self, feature: str, method: str, value=None):
        self.feature = feature
        self.method = method
        self.value = value

class RescaleMethod(Enum):
    MINMAX = "MINMAX"
    AVGSTD = "AVGSTD"

class ColumnType(Enum):
    CATEGORICAL = "categorical"
    TEXT = "text"
    NUMERICAL = "numerical"

class Algorithm(Enum):
    KNN = "KNN"
    DecisionTree = "DecisionTree"
    RandomForest = "RandomForest"
    NaiveBayes = "NaiveBayes"

class RescaleFeature:
    def __init__(self, feature: str, method: str):
        self.feature = feature
        self.method = method

class DropInstance:
    EQUAL = "=="
    LESS = "<"
    MORE = ">"
    LESS_EQ = "<="
    MORE_EQ = ">="
    UNEQUAL = "!="
    def __init__(self, column, comparison, value):
        self.column = column
        self.comparison = comparison
        self.value = value

class NlpEmoji:
    def __init__(
            self,
            column: str,
            delete: bool,
            metrics: set[str],
            bin: bool
        ) -> None:
        self.column = column
        self.delete = delete
        self.metrics = metrics
        self.bin = bin

# Clase básica para cualquier algoritmo
class AlgorithmConfig:
    def __init__(self,
                impute: bool,
                drop: bool,
                preprocess_categorical: bool,
                preprocess_text: bool,
                preprocess_numerical: bool,
                rescale: bool,
                maximize_value: str,
                maximize_average: str,
                name: str
        ) -> None:
        self.impute = impute
        self.drop = drop
        self.preprocess_categorical = preprocess_categorical
        self.preprocess_text = preprocess_text
        self.preprocess_numerical = preprocess_numerical
        self.rescale = rescale
        self.maximize_value = maximize_value
        self.maximize_average = maximize_average
        self.name = name

# Hiper-hiperparámetros del algoritmo KNN
class KnnConfig(AlgorithmConfig):
    def __init__(self,
                mink: int,
                maxk: int,
                stepk: int,
                minp: int,
                maxp: int,
                weights: list[str],
                impute: bool,
                drop: bool,
                preprocess_categorical: bool,
                preprocess_text: bool,
                preprocess_numerical: bool,
                rescale: bool,
                maximize_value: str,
                maximize_average: str,
                name: str
            ):
        self.mink = mink
        self.maxk = maxk
        self.stepk = stepk
        self.minp = minp
        self.maxp = maxp
        self.weights = weights
        super().__init__(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale,
                maximize_value,
                maximize_average,
                name
            )

# Hiper-hiperparámetros del algoritmo decision tree
class DecisionTreeConfig(AlgorithmConfig):
    def __init__(self,
                impute: bool,
                drop: bool,
                preprocess_categorical: bool,
                preprocess_text: bool,
                preprocess_numerical: bool,
                rescale: bool,
                criterios: list[str],
                splitters: list[str],
                min_maxDepth: int,
                max_maxDepth: int,
                min_minSamplesLeaf: int,
                max_minSamplesLeaf: int,
                maximize_value: str,
                maximize_average: str,
                name: str,
                binning: bool
            ):
        self.criterios = criterios
        self.splitters = splitters
        self.min_maxDepth = min_maxDepth
        self.max_maxDepth = max_maxDepth
        self.min_minSamplesLeaf = min_minSamplesLeaf
        self.max_minSamplesLeaf = max_minSamplesLeaf
        self.binning = binning
        super().__init__(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale,
                maximize_value,
                maximize_average,
                name)

# Hiper-hiperparámetros del algoritmo random forest
class RandomForestConfig(AlgorithmConfig):
    def __init__(self,
                impute: bool,
                drop: bool,
                preprocess_categorical: bool,
                preprocess_text: bool,
                preprocess_numerical: bool,
                rescale: bool,
                maximize_value: str,
                maximize_average: str,
                criterios: list[str],
                splitters: list[str],
                min_maxDepth: int,
                max_maxDepth: int,
                min_minSamplesLeaf: int,
                max_minSamplesLeaf: int,
                name: str,
                binning: bool,
                n_estimators: int
            ):
        self.criterios = criterios
        self.splitters = splitters
        self.min_maxDepth = min_maxDepth
        self.max_maxDepth = max_maxDepth
        self.min_minSamplesLeaf = min_minSamplesLeaf
        self.max_minSamplesLeaf = max_minSamplesLeaf
        self.binning = binning
        self.n_estimators = n_estimators
        super().__init__(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale,
                maximize_value,
                maximize_average,
                name)

# Hiper-hiperparámetros del algoritmo Naïve Bayes
class NaiveBayesConfig(AlgorithmConfig):
    GAUSSIAN = "gaussian"
    CATEGORICAL = "categorical"
    MULTINOMIAL = "multinomial"

    def __init__(self,
                impute: bool,
                drop: bool,
                preprocess_categorical: bool,
                preprocess_text: bool,
                preprocess_numerical: bool,
                rescale: bool,
                maximize_value: str,
                maximize_average: str,
                name: str,
                type_: str,
                binning: bool,
                min_alpha,
                max_alpha,
                step_alpha
            ):
        self.type_ = type_
        self.binning = binning
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        super().__init__(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale,
                maximize_value,
                maximize_average,
                name)

def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)

def punto_coma(x: str):
    return x.replace(",", ".")

def get_emoji_metric(inp: str, metric: str, delete_emojis: bool):
    value = 0
    emojis_found = set()
    for char in inp:
        if char in EMOJI_SENTIMENT_DICT:
            new_value = get_emoji_sentiment_rank(char)[metric]
            value += new_value
            emojis_found.add(char)
    if delete_emojis:
        for emoji in emojis_found:
            inp = inp.replace(emoji, "")

    return (inp, value)

def preprocess(
        ml_dataset,
        columns,
        punto_a_coma_columns,
        categorical_features,
        text_features,
        numerical_features,
        predict_column,
        target_map,
        preprocess_numerical,
        binning,
        binning_features,
        binning_arg,
        tf_idf_columns,
        drop_instances: list[DropInstance],
        nlp_emoji: list[NlpEmoji],
        stop_words: StopWords,
        testing: bool,
        tf_idf_pickle_name: str,
        bin_tfidf: bool
):
    """preprocesa los datos, tanto como para entrenar el modelo como para usarlo"""

    ml_dataset = ml_dataset[columns]

    for feature in categorical_features:
        # transforma las categóricas a unicode
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        # transforma las de texto a unicode
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in punto_a_coma_columns:
        # Cambia las , en esta columna a .
        ml_dataset[feature] = ml_dataset[feature].apply(punto_coma)

    for drop in drop_instances:
        if drop.comparison == DropInstance.EQUAL:
            ml_dataset.drop(ml_dataset[ml_dataset[drop.column] == drop.value].index, inplace = True)
        elif drop.comparison == DropInstance.LESS:
            ml_dataset.drop(ml_dataset[ml_dataset[drop.column] < drop.value].index, inplace = True)
        elif drop.comparison == DropInstance.LESS_EQ:
            ml_dataset.drop(ml_dataset[ml_dataset[drop.column] <= drop.value].index, inplace = True)
        elif drop.comparison == DropInstance.MORE:
            ml_dataset.drop(ml_dataset[ml_dataset[drop.column] > drop.value].index, inplace = True)
        elif drop.comparison == DropInstance.MORE_EQ:
            ml_dataset.drop(ml_dataset[ml_dataset[drop.column] >= drop.value].index, inplace = True)
        elif drop.comparison == DropInstance.UNEQUAL:
            ml_dataset.drop(ml_dataset[ml_dataset[drop.column] != drop.value].index, inplace = True)
        else:
            print("Error inesperado")
            exit(1)

    for nlp_emoji_col in nlp_emoji:
        column = nlp_emoji_col.column
        for metric in nlp_emoji_col.metrics:
            new_column = f"{column}_{metric}_emojirank"
            print(new_column)

            ml_dataset[new_column] = ml_dataset[column].apply(lambda x: get_emoji_metric(x, metric, False)[1]) # Crea nueva columna

            if nlp_emoji_col.bin:
                binning_features.append(new_column)

        if nlp_emoji_col.delete:
            ml_dataset[column] = ml_dataset[column].apply(lambda x: get_emoji_metric(x, "positive", True)[0]) # Borra los emojis de la anterior
        

    for (index, feature) in enumerate(tf_idf_columns):
        if testing:
            tf = pickle.load(open(tf_idf_pickle_name, 'rb'))
        else:
            if stop_words.any_change():
                if stop_words.start_from_english:
                    final_stop_words = text.ENGLISH_STOP_WORDS
                else:
                    final_stop_words = frozenset()

                final_stop_words = final_stop_words.union(stop_words.add_words)
                final_stop_words = final_stop_words.difference(stop_words.remove_words)
                final_stop_words = list(final_stop_words)
                print(final_stop_words)
                #tf = TfidfVectorizer(stop_words=final_stop_words, tokenizer=LemmaTokenizer()) # type: ignore
                tf = TfidfVectorizer(stop_words=final_stop_words)
            else:
                #tf = TfidfVectorizer(tokenizer=LemmaTokenizer()) # type: ignore
                tf = TfidfVectorizer()
            tf.fit(ml_dataset[feature])
            pickle.dump(tf, open(tf_idf_pickle_name,"wb"))

        """
        print("AAA")
        print(tf.vocabulary_)
        print("AAA")
        print(tf.stop_words_)
        print("AAA")
        """
        text_tf = tf.transform(ml_dataset[feature])
        #text_tf = pd.DataFrame(text_tf.toarray().transpose(),
        #           index=tf.get_feature_names_out())
        for (index2, column) in enumerate(text_tf.transpose()):
            column_name = f"tf_idf_{index}_{index2}"
            ml_dataset[column_name] = column.toarray()[0]
            if bin_tfidf:
                binning_features.append(column_name)
        #text_tf.append(ml_dataset[TARGET])
        #feature_sets.append(text_tf)
        del ml_dataset[feature]

    if preprocess_numerical:
        for feature in numerical_features:
            # M8[ns] = nanosegundos desde 1 jan 1970
            if (ml_dataset[feature].dtype == np.dtype("M8[ns]")\
                or (\
                    hasattr(ml_dataset[feature].dtype, "base")\
                    and ml_dataset[feature].dtype.base == np.dtype("M8[ns]")\
                )):

                # si es fecha, convierte a fecha normal
                #ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
                print("Hay fechas, pon el preprocess_numerical a false")
                exit(1)
            else:
                # si no, lo convierte a double
                ml_dataset[feature] = ml_dataset[feature].astype('double')
    else:
        for feature in numerical_features:
            ml_dataset[feature] = ml_dataset[feature].astype('double')
    
    binnings_save = {}
    if binning:
        for feature in binning_features:
            if isinstance(binning_arg, int):
                # binning_arg es el número de bins
                _min = ml_dataset[feature].min()
                _max = ml_dataset[feature].max()

                step = (_max - _min) / binning_arg

                bins = []
                for bin_incr in range(0, binning_arg):
                    bins.append(_min + bin_incr*step)

                binnings_save[feature] = bins
            else:
                bins = binning_arg[feature]
            
            ml_dataset[feature] = ml_dataset[feature].apply(lambda x: fit_bin(bins, x))
    
    if len(binnings_save.items()) != 0:
        # guardarlos en un archivo
        with open("last_binnings.json", "w") as f:
            json.dump(binnings_save, f)

    #convierte esta columna en un int de python, independientemente de su tipo original y lo mete en una nueva columna
    ml_dataset[TARGET] = ml_dataset[predict_column].map(str).map(target_map)
    del ml_dataset[predict_column]

    ml_dataset = ml_dataset[~ml_dataset[TARGET].isnull()]
    return ml_dataset

def fit_bin(bins, num):
    last_index = -1
    for index, bin in enumerate(bins):
        last_index = index
        if num < bin:
            return index

    return last_index + 1

TARGET = "__target__"
def run(
        ml_dataset,
        columns,
        punto_a_coma_columns,
        categorical_features,
        text_features,
        numerical_features,
        predict_column,
        target_map,
        drop_rows_when_missing,
        impute_when_missing: list[ImputeFeature],
        test_size,
        rescale_features: list[RescaleFeature],
        undersampling_ratio,
        oversampling_ratio,
        algorithm: AlgorithmConfig,
        binning_columns,
        binning_arg,
        tf_idf_columns,
        drop_after_preprocess_columns,
        drop_instances,
        nlp_emoji,
        drop_instances_train: list[DropInstance],
        drop_instances_dev: list[DropInstance],
        stop_words: StopWords,
        tf_idf_pickle_name: str,
        bin_tfidf: bool
    ) -> str:
    """dada una configuración, entrena el mejor modelo de un algoritmo"""

    if isinstance(algorithm, DecisionTreeConfig):
        binning = algorithm.binning
    elif isinstance(algorithm, RandomForestConfig):
        binning = algorithm.binning
    elif isinstance(algorithm, NaiveBayesConfig):
        binning = algorithm.binning
    else:
        binning = False

    ml_dataset = preprocess(
        ml_dataset,
        columns,
        punto_a_coma_columns,
        categorical_features,
        text_features,
        numerical_features,
        predict_column,
        target_map,
        algorithm.preprocess_numerical,
        binning,
        binning_columns,
        binning_arg,
        tf_idf_columns,
        drop_instances,
        nlp_emoji,
        stop_words,
        False,
        tf_idf_pickle_name,
        bin_tfidf
    )
    strat = ml_dataset[[TARGET]]
    (train, test) = train_test_split(ml_dataset,test_size=test_size,random_state=42,stratify=strat)
    if algorithm.drop:
        # Explica lo que se hace en este paso
        for feature in drop_rows_when_missing:
            # se queda solo con las instancias que no tienen valores nullos
            train = train[train[feature].notnull()]
            test = test[test[feature].notnull()]
            print('Dropped missing records in %s' % feature)

    if algorithm.impute:
        # Explica lo que se hace en este paso
        for feature in impute_when_missing:
            # Calcula el valor para reemplazar los na
            v = None;
            if feature.method == ImputeMethod.MEAN:
                v = train[feature.feature].mean() # type: ignore
            elif feature.method == ImputeMethod.MEDIAN:
                v = train[feature.feature].median() # type: ignore
            elif feature.method == ImputeMethod.CREATE_CATEGORY:
                v = 'NULL_CATEGORY'
            elif feature.method == ImputeMethod.MODE:
                v = train[feature.feature].value_counts().index[0] # type: ignore
            elif feature.method == ImputeMethod.CONSTANT:
                v = feature.value

            if v is None:
                print("No hay v")
                exit(1)

            # Llena los na con el valor calculado
            train[feature.feature] = train[feature.feature].fillna(v) # type: ignore
            test[feature.feature] = test[feature.feature].fillna(v) # type: ignore
            print('Imputed missing values in feature %s with value %s' % (feature.feature, coerce_to_unicode(v)))

    if algorithm.rescale:
        for feature in rescale_features:
            if feature.method == RescaleMethod.MINMAX:
                _min = train[feature.feature].min() # type: ignore
                _max = train[feature.feature].max() # type: ignore
                scale = _max - _min
                shift = _min
            elif feature.method == RescaleMethod.AVGSTD:
                shift = train[feature.feature].mean() # type: ignore
                scale = train[feature.feature].std() # type: ignore
            else:
                print("Bad rescale method")
                exit(1)
            if scale == 0.:
                del train[feature.feature] # type: ignore
                del test[feature.feature] # type: ignore
                print('Feature %s was dropped because it has no variance' % feature.feature)
            else:
                print('Rescaled %s' % feature.feature)
                train[feature.feature] = (train[feature.feature] - shift).astype(np.float64) / scale # type: ignore
                test[feature.feature] = (test[feature.feature] - shift).astype(np.float64) / scale # type: ignore

    for drop_instance in drop_instances_train:
        if drop_instance.comparison == DropInstance.EQUAL:
            train.drop(train[train[drop_instance.column] == drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.LESS:
            train.drop(train[train[drop_instance.column] < drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.LESS_EQ:
            train.drop(train[train[drop_instance.column] <= drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.MORE:
            train.drop(train[train[drop_instance.column] > drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.MORE_EQ:
            train.drop(train[train[drop_instance.column] >= drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.UNEQUAL:
            train.drop(train[train[drop_instance.column] != drop_instance.value].index, inplace = True)
        else:
            print("Error inesperado")
            exit(1)
    for drop_instance in drop_instances_dev:
        if drop_instance.comparison == DropInstance.EQUAL:
            test.drop(test[test[drop_instance.column] == drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.LESS:
            test.drop(test[test[drop_instance.column] < drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.LESS_EQ:
            test.drop(test[test[drop_instance.column] <= drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.MORE:
            test.drop(test[test[drop_instance.column] > drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.MORE_EQ:
            test.drop(test[test[drop_instance.column] >= drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == DropInstance.UNEQUAL:
            test.drop(test[test[drop_instance.column] != drop_instance.value].index, inplace = True)
        else:
            print("Error inesperado")
            exit(1)

    for column in drop_after_preprocess_columns:
        print(f"dropping {column}")
        test.drop(column, axis=1, inplace=True)
        train.drop(column, axis=1, inplace=True)

    trainX = train.drop(TARGET, axis=1) # type: ignore
    #trainY = train[TARGET]

    testX = test.drop(TARGET, axis=1) # type: ignore
    #testY = test[TARGET]

    trainY = np.array(train[TARGET]) # type: ignore
    testY = np.array(test[TARGET]) # type: ignore


    print("SAMPLING")
    # Explica lo que se hace en este paso
    if undersampling_ratio is not None:
        # Hace un undersample (elimina instancias de las clases más representadas)
        undersample = RandomUnderSampler(sampling_strategy=undersampling_ratio)#la mayoria va a estar representada el doble de veces

        trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY) # type: ignore
        testXUnder,testYUnder = undersample.fit_resample(testX, testY) # type: ignore

        trainX,trainY = trainXUnder, trainYUnder
        testX,testY = testXUnder, testYUnder
    
    if oversampling_ratio is not None:
        # Hace un undersample (crea instancias de las clases menos representadas)
        oversample = RandomOverSampler(sampling_strategy=oversampling_ratio)
        trainXUnder,trainYUnder = oversample.fit_resample(trainX,trainY) # type: ignore
        testXUnder,testYUnder = oversample.fit_resample(testX, testY) # type: ignore

        trainX,trainY = trainXUnder, trainYUnder
        testX,testY = testXUnder, testYUnder

    # Según el algoritmo, ejecuta su función para entrenar el modelo
    # Con barrido de hiperparámetros
    if isinstance(algorithm, KnnConfig):
        algorithm = cast(KnnConfig, algorithm)
        return run_knn(algorithm, trainX, trainY, testX, testY, target_map, test)
    elif isinstance(algorithm, RandomForestConfig):
        algorithm = cast(RandomForestConfig, algorithm)
        print("RANDOM FOREST")
        return run_random_forest(algorithm, trainX, trainY, testX, testY, target_map, test)
    elif isinstance(algorithm, DecisionTreeConfig):
        algorithm = cast(DecisionTreeConfig, algorithm)
        return run_decision_tree(algorithm, trainX, trainY, testX, testY, target_map, test)
    elif isinstance(algorithm, NaiveBayesConfig):
        algorithm = cast(NaiveBayesConfig, algorithm)
        return run_naive_bayes(algorithm, trainX, trainY, testX, testY, target_map, test)
    else:
        print("No se ha elegido algoritmo")
        exit(1)

def run_knn(
    algorithm: KnnConfig,
    trainX,
    trainY,
    testX,
    testY,
    target_map,
    test
):
    best_alg = None
    best_alg_params = (None, None, None)
    best_score = 0.0
    scores = []
    for k in range(algorithm.mink, algorithm.maxk+1, algorithm.stepk):
        for p in range(algorithm.minp, algorithm.maxp+1):
            for weight in algorithm.weights:
                clf = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weight, # type: ignore
                    algorithm='auto',
                    leaf_size=30,
                    p=p
                )

                clf.class_weight = "balanced" # type: ignore

                # Explica lo que se hace en este paso

                clf.fit(trainX, trainY)


                # Build up our result dataset

                # The model is now trained, we can apply it to our test set:

                predictions = clf.predict(testX)
                probas = clf.predict_proba(testX)
                score = evaluate(predictions, probas, testX, testY, target_map, test, algorithm.maximize_value, algorithm.maximize_average)
                if score > best_score:
                    best_score = score
                    best_alg = clf
                    best_alg_params = (k, p, weight)
                scores.append((f"{k}, {p}, {weight}", score))
    pickle.dump(best_alg, open(algorithm.name,"wb"))
    (k, p, weight) = best_alg_params
    print(f"Mejor knn: k={k}, p={p}, weight={weight}")
    print(f"Puntuación: {best_score}")
    print(f"Guardando knn {algorithm.name}")
    def extract_key(x):
        (_, score) = x
        return score
    scores.sort(key=extract_key, reverse=True)
    info = "k\tp\tweight\tscore\n"
    for (stringargs, score) in scores:
        info += stringargs + f", {score}\n"

    return info

def run_random_forest(
    algorithm: RandomForestConfig,
    trainX,
    trainY,
    testX,
    testY,
    target_map,
    test
):
    best_alg = None
    best_alg_params = (None, None)
    best_score = 0.0
    scores = []
    iteration = 0
    for max_depth in range(algorithm.min_maxDepth, algorithm.max_maxDepth+1):
        for min_samples_leaf in range(algorithm.min_minSamplesLeaf, algorithm.max_minSamplesLeaf+1):
            clf = RandomForestClassifier(n_estimators=algorithm.n_estimators,
                random_state=1337,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf
            )
            iteration += 1
            print(f"{iteration}, {max_depth}, {min_samples_leaf}")

            clf.class_weight = "balanced" # type: ignore

            # Explica lo que se hace en este paso

            clf.fit(trainX, trainY)


            # Build up our result dataset

            # The model is now trained, we can apply it to our test set:

            predictions = clf.predict(testX)
            probas = clf.predict_proba(testX)
            score = evaluate(predictions, probas, testX, testY, target_map, test, algorithm.maximize_value, algorithm.maximize_average)
            if score > best_score:
                best_score = score
                best_alg = clf
                best_alg_params = (max_depth, min_samples_leaf)
            scores.append((f"{max_depth}, {min_samples_leaf}", score))
    pickle.dump(best_alg, open(algorithm.name,"wb"))
    (max_depth, min_samples_leaf) = best_alg_params
    print(f"Mejor random forest: max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")
    print(f"Puntuación: {best_score}")
    print(f"Guardando random forest {algorithm.name}")

    def extract_key(x):
        (_, score) = x
        return score
    scores.sort(key=extract_key, reverse=True)
    info = "max_depth\tmin_samples_leaf\tscore\n"
    for (stringargs, score) in scores:
        info += stringargs + f", {score}\n"

    return info

def run_decision_tree(
    algorithm: DecisionTreeConfig,
    trainX,
    trainY,
    testX,
    testY,
    target_map,
    test
):
    best_alg = None
    best_alg_params = (None, None, None, None)
    best_score = 0.0
    scores = []
    for criterio in algorithm.criterios:
        for splitter in algorithm.splitters:
            for max_depth in range(algorithm.min_maxDepth, algorithm.max_maxDepth+1):
                for min_samples_leaf in range(algorithm.min_minSamplesLeaf, algorithm.max_minSamplesLeaf+1):
                    clf = DecisionTreeClassifier(
                        random_state=1337,
                        criterion = criterio, # type: ignore
                        splitter = splitter, # type: ignore
                        max_depth = max_depth,
                        min_samples_leaf = min_samples_leaf
                    )

                    clf.class_weight = "balanced" # type: ignore

                    # Explica lo que se hace en este paso

                    clf.fit(trainX, trainY)


                    # Build up our result dataset

                    # The model is now trained, we can apply it to our test set:

                    predictions = clf.predict(testX)
                    probas = clf.predict_proba(testX)
                    score = evaluate(predictions, probas, testX, testY, target_map, test, algorithm.maximize_value, algorithm.maximize_average)
                    if score > best_score:
                        best_score = score
                        best_alg = clf
                        best_alg_params = (criterio, splitter, max_depth, min_samples_leaf)
                    scores.append((f"{criterio}, {splitter}, {max_depth}, {min_samples_leaf}", score))
    pickle.dump(best_alg, open(algorithm.name,"wb"))
    (criterio, splitter, max_depth, min_samples_leaf) = best_alg_params
    print(f"Mejor decision tree: criterio={criterio}, splitter={splitter}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")
    print(f"Puntuación: {best_score}")
    print(f"Guardando decision tree {algorithm.name}")

    def extract_key(x):
        (_, score) = x
        return score
    scores.sort(key=extract_key, reverse=True)
    info = "criterio\tsplitter\tmax_depth\tmin_samples_leaf\tscore\n"
    for (stringargs, score) in scores:
        info += stringargs + f", {score}\n"

    return info

def run_naive_bayes(
    algorithm: NaiveBayesConfig,
    trainX,
    trainY,
    testX,
    testY,
    target_map,
    test
):
    best_alg = None
    best_alg_params = 0
    best_score = 0.0
    scores = []
    alpha = algorithm.min_alpha
    while True:
        if alpha > max_alpha:
            break

        if algorithm.type_ == NaiveBayesConfig.GAUSSIAN:
            clf = GaussianNB(var_smoothing=alpha)
        elif algorithm.type_ == NaiveBayesConfig.CATEGORICAL:
            clf = CategoricalNB(alpha=alpha)
        elif algorithm.type_ == NaiveBayesConfig.MULTINOMIAL:
            clf = MultinomialNB(alpha=alpha)
        else:
            print(f"Algoritmo naive bayes no reconocido {algorithm.type_}")
            exit(1)

        clf.fit(trainX, trainY)

        predictions = clf.predict(testX)
        probas = clf.predict_proba(testX)
        score = evaluate(predictions, probas, testX, testY, target_map, test, algorithm.maximize_value, algorithm.maximize_average)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_alg = clf
            best_alg_params = alpha
        
        alpha += algorithm.step_alpha
    print(scores)
    pickle.dump(best_alg, open(algorithm.name,"wb"))
    print(f"Mejor naive bayes")
    print(f"Puntuación: {best_score}, alpha={best_alg_params}")
    print(f"Guardando naive bayes {algorithm.name}")

    return ""

def evaluate(predictions, probas, testX, testY, target_map, test, maximize_value, maximize_average) -> float:
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    cols = [
        u'probability_of_value_%s' % label
        for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    ]
    probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

# Build scored dataset
    results_test = testX.join(predictions, how='left')
    results_test = results_test.join(probabilities, how='left')
    results_test = results_test.join(test[TARGET], how='left') # type: ignore
    results_test = results_test.rename(columns= {TARGET: 'TARGET'})

    if NEGATIVE_EVALUATION:
        print("fscore")
        print(f1_score(testY, predictions, average=None)[2])
        print("precision")
        print(precision_score(testY, predictions, average=None)[2])
        print("recall")
        print(recall_score(testY, predictions, average=None)[2])
        if maximize_value == MaximizeValue.FSCORE:
            print(f1_score(testY, predictions, average=None))
            return f1_score(testY, predictions, average=None)[2] # type: ignore
        elif maximize_value == MaximizeValue.PRECISION:
            return precision_score(testY, predictions, average=None)[2] # type: ignore
        elif maximize_value == MaximizeValue.RECALL:
            return recall_score(testY, predictions, average=None)[2] # type: ignore
        elif maximize_value == MaximizeValue.ACCURACY:
            return accuracy_score(testY, predictions, average=None)[2] # type: ignore
        else:
            print("método de maximizar no disponible")
            exit(1)
    else:
        print("fscore")
        print(f1_score(testY, predictions, average=maximize_average))
        print("precision")
        print(precision_score(testY, predictions, average=maximize_average))
        print("recall")
        print(recall_score(testY, predictions, average=maximize_average))
        if maximize_value == MaximizeValue.FSCORE:
            return f1_score(testY, predictions, average=maximize_average) # type: ignore
        elif maximize_value == MaximizeValue.PRECISION:
            return precision_score(testY, predictions, average=maximize_average) # type: ignore
        elif maximize_value == MaximizeValue.RECALL:
            return recall_score(testY, predictions, average=maximize_average) # type: ignore
        elif maximize_value == MaximizeValue.ACCURACY:
            return accuracy_score(testY, predictions, average=maximize_average) # type: ignore
        else:
            print("método de maximizar no disponible")
            exit(1)

T = TypeVar("T")
def get_att_default(dictionary: dict[str, T], att: str, default: T) -> T :
    try:
        return dictionary[att]
    except:
        return default

def json_bool(j_bool: str) -> bool:
    if j_bool == "true":
        return True
    elif j_bool == "false":
        return False
    else:
        print("Has dado un valor no booleano a un booleano en el json")
        exit(1)

def get_config(config):
    INPUT_FILE = config["input_file"]
    ml_dataset = pd.read_csv(INPUT_FILE)

    COLUMNS_CONFIG = config["columns"]
    continuar = True
    for column in COLUMNS_CONFIG["columns"]:
        if column not in ml_dataset.columns:
            print(f"La columna {column} no aparece en el fichero")
            continuar = False
    if not continuar:
        exit(1)

    COLUMNS = []
    if COLUMNS_CONFIG["list_type"] == "excluding":
        for column in ml_dataset.columns:
            if column not in COLUMNS_CONFIG["columns"]:
                COLUMNS.append(column)
    elif COLUMNS_CONFIG["list_type"] == "including":
        COLUMNS = COLUMNS_CONFIG["columns"]
    else:
        print("columns.list_type solo puede ser excluding o including")
        exit(1)

    PUNTO_A_COMA_COLUMNS = get_att_default(config, "punto_a_coma_columns", [])
    for column in PUNTO_A_COMA_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de punto_a_coma_columns no está en columns")
            exit(1)
    CATEGORICAL_COLUMNS = get_att_default(config, "categorical_columns", [])
    for column in CATEGORICAL_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de categorical_columns no está en columns")
            exit(1)
    TEXT_COLUMNS = get_att_default(config, "text_columns", [])
    for column in TEXT_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de text_columns no está en columns")
            exit(1)
    NUMERICAL_COLUMNS = get_att_default(config, "numerical_columns", [])
    for column in NUMERICAL_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de numerical_columns no está en columns")
            exit(1)
    BINNING_COLUMNS = get_att_default(config, "binning_columns", [])
    for column in BINNING_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de binning_columns no está en columns")
            exit(1)
    TF_IDF_COLUMNS = get_att_default(config, "tf_idf_columns", [])
    for column in TF_IDF_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de tf_idf_columns no está en columns")
            exit(1)
    DROP_AFTER_PREPROCESS_COLUMNS = get_att_default(config, "drop_after_preprocess_columns", [])
    for column in DROP_AFTER_PREPROCESS_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de drop_after_preprocess_columns no está en columns")
            exit(1)
    NLP_EMOJI = get_att_default(config, "nlp_emoji", [])
    nlp_emoji = []
    for item in NLP_EMOJI:
        column = item["column"]
        if column not in COLUMNS:
            print(f"{column} de nlp_emoji no está en columns")
            exit(1)
        delete = json_bool(item["delete_emojis"])
        metrics_list = item["metrics"]
        metrics = set() # Para asegurarse de que no hay repeticiones
        for metric in metrics_list:
            metrics.add(metric)

        bin = json_bool(item["bin"])
        nlp_emoji.append(NlpEmoji(column, delete, metrics, bin))
    try:
        rest_columns = config["rest_columns"]
        if rest_columns == ColumnType.CATEGORICAL:
            for column in COLUMNS:
                if (
                    column not in CATEGORICAL_COLUMNS
                    and column not in TEXT_COLUMNS
                    and column not in NUMERICAL_COLUMNS
                ):
                    CATEGORICAL_COLUMNS.append(column)

        if rest_columns == ColumnType.TEXT:
            for column in COLUMNS:
                if (
                    column not in CATEGORICAL_COLUMNS
                    and column not in TEXT_COLUMNS
                    and column not in NUMERICAL_COLUMNS
                ):
                    TEXT_COLUMNS.append(column)
        if rest_columns == ColumnType.NUMERICAL:
            for column in COLUMNS:
                if (
                    column not in CATEGORICAL_COLUMNS
                    and column not in TEXT_COLUMNS
                    and column not in NUMERICAL_COLUMNS
                ):
                    NUMERICAL_COLUMNS.append(column)
        else:
            print(f"Tipo de columna {rest_columns} no es válido")
            exit(1)
    except:
        pass # rest_columns no está definido
    PREDICT_COLUMN = config["predict_column"]

    stop_words = get_att_default(config, "stop_words", None)
    if stop_words is not None:
        start_from_english = json_bool(get_att_default(stop_words, "start_from_english", "false"))
        add_words = get_att_default(stop_words, "add_words", [])
        remove_words = get_att_default(stop_words, "remove_words", [])
        stop_words = StopWords(start_from_english, add_words, remove_words)
    else:
        stop_words = StopWords(False, [], [])

    DROP_INSTANCES = get_att_default(config, "drop_instance", [])
    drop_instances = []
    possible_comparisons = [DropInstance.EQUAL, DropInstance.LESS, DropInstance.MORE, DropInstance.LESS_EQ, DropInstance.MORE_EQ, DropInstance.UNEQUAL]
    for item in DROP_INSTANCES:
        try:
            columna = item["columna"]
        except Exception:
            print("No se ha especificado columna en un drop_instance")
            exit(1)
        if item["comparison"] not in possible_comparisons:
            print(f"Comparación incorrecta en la columna {columna}")
            exit(1)

        drop_instances.append(DropInstance(
            columna, item["comparison"], item["value"]
        ))

        if columna not in COLUMNS:
            print(f"{columna} de drop_instances no está en columns")
            exit(1)

    TARGET_MAP = config["target_map"]
    DROP_ROWS_WHEN_MISSING = config["drop_rows_when_missing"]
    for column in DROP_ROWS_WHEN_MISSING:
        if column not in COLUMNS:
            print(f"{column} de drop_rows_when_missing no está en columns")
            exit(1)
    IMPUTE_WHEN_MISSING = []
    for item in config["impute_when_missing"]:
        value = None
        column = item["feature"]
        try:
            if item["method"] == "MEAN":
                method = ImputeMethod.MEAN
            elif item["method"] == "MEDIAN":
                method = ImputeMethod.MEDIAN
            elif item["method"] == "CREATE_CATEGORY":
                method = ImputeMethod.CREATE_CATEGORY
            else:
                raise Exception("ImputeMethod no valido")
        except KeyError:
            try:
                value = item["constant"]
                method = ImputeMethod.CONSTANT
            except KeyError:
                raise Exception(f"impute de la columna {column} sin método")

        if column not in COLUMNS:
            print(f"{column} de impute_when_missing no está en columns")
            exit(1)
        IMPUTE_WHEN_MISSING.append(ImputeFeature(column, method, value))
    TEST_SIZE = config["test_size"]
    RESCALE_FEATURES = []
    for item in config["rescale_features"]:
        if item["method"] == "MINMAX":
            method = RescaleMethod.MINMAX
        elif item["method"] == "AVGSTD":
            method = RescaleMethod.AVGSTD
        else:
            raise Exception("RescaleMethod no valido")
        
        column = item["feature"]
        if column not in COLUMNS:
            print(f"{column} de drop_rows_when_missing no está en columns")
            exit(1)
        RESCALE_FEATURES.append(RescaleFeature(column, method))

    DROP_INSTANCES_TRAIN = get_att_default(config, "drop_from_train", [])
    drop_instances_train = []
    for item in DROP_INSTANCES_TRAIN:
        try:
            columna = item["column"]
        except Exception:
            print("No se ha especificado columna en un drop_from_train")
            exit(1)
        if item["comparison"] not in possible_comparisons:
            print(f"Comparación incorrecta en la columna {columna}")
            exit(1)

        drop_instances_train.append(DropInstance(
            columna, item["comparison"], item["value"]
        ))

        if columna not in COLUMNS:
            print(f"{columna} de drop_from_train no está en columns")
            exit(1)
    
    DROP_INSTANCES_DEV = get_att_default(config, "drop_from_dev", [])
    drop_instances_dev = []
    for item in DROP_INSTANCES_DEV:
        try:
            columna = item["column"]
        except Exception:
            print("No se ha especificado columna en un drop_from_dev")
            exit(1)
        if item["comparison"] not in possible_comparisons:
            print(f"Comparación incorrecta en la columna {columna}")
            exit(1)

        drop_instances_dev.append(DropInstance(
            columna, item["comparison"], item["value"]
        ))

        if columna not in COLUMNS:
            print(f"{columna} de drop_from_dev no está en columns")
            exit(1)

    DROP_INSTANCES_TEST = get_att_default(config, "drop_from_test", [])
    drop_instances_test = []
    for item in DROP_INSTANCES_TEST:
        try:
            columna = item["column"]
        except Exception:
            print("No se ha especificado columna en un drop_from_test")
            exit(1)
        if item["comparison"] not in possible_comparisons:
            print(f"Comparación incorrecta en la columna {columna}")
            exit(1)

        drop_instances_test.append(DropInstance(
            columna, item["comparison"], item["value"]
        ))

        if columna not in COLUMNS:
            print(f"{columna} de drop_from_test no está en columns")
            exit(1)

    try:
        UNDERSAMPLING_RATIO = config["undersampling_ratio"]
    except KeyError:
        UNDERSAMPLING_RATIO = None

    try:
        OVERSAMPLING_RATIO = config["oversampling_ratio"]
    except KeyError:
        OVERSAMPLING_RATIO = None
    
    algorithms = config["algorithms"]

    TF_IDF_PICKLE_NAME = get_att_default(config, "tf_idf_pickle_name", "default_tf_save.tfidf")
    BIN_TFIDF = json_bool(get_att_default(config, "bin_tfidf", "false"))

    return (
        algorithms,
        ml_dataset,
        COLUMNS,
        PUNTO_A_COMA_COLUMNS,
        CATEGORICAL_COLUMNS,
        TEXT_COLUMNS,
        NUMERICAL_COLUMNS,
        PREDICT_COLUMN,
        TARGET_MAP,
        DROP_ROWS_WHEN_MISSING,
        IMPUTE_WHEN_MISSING,
        TEST_SIZE,
        RESCALE_FEATURES,
        UNDERSAMPLING_RATIO,
        OVERSAMPLING_RATIO,
        BINNING_COLUMNS,
        TF_IDF_COLUMNS,
        DROP_AFTER_PREPROCESS_COLUMNS,
        drop_instances,
        nlp_emoji,
        drop_instances_dev,
        drop_instances_train,
        drop_instances_test,
        stop_words,
        TF_IDF_PICKLE_NAME,
        BIN_TFIDF
    )

if __name__ == "__main__":
    if sys.version_info < (3, 0):
        print("Estás usando python 2, usa el 3")
        exit(1)

    if len(sys.argv) != 2:
        print("Usage: python plantilla_SAD.py config.json")
        exit(1)
    with open(sys.argv[1]) as f:
        config = json.load(f)
        with open(config["$schema"]) as f:
            schema = json.load(f)
        try:
            jsonschema.validate(config, schema)
        except Exception as e:
            print(e)

    (
        algorithms,
        ml_dataset,
        COLUMNS,
        PUNTO_A_COMA_COLUMNS,
        CATEGORICAL_COLUMNS,
        TEXT_COLUMNS,
        NUMERICAL_COLUMNS,
        PREDICT_COLUMN,
        TARGET_MAP,
        DROP_ROWS_WHEN_MISSING,
        IMPUTE_WHEN_MISSING,
        TEST_SIZE,
        RESCALE_FEATURES,
        UNDERSAMPLING_RATIO,
        OVERSAMPLING_RATIO,
        BINNING_COLUMNS,
        TF_IDF_COLUMNS,
        DROP_AFTER_PREPROCESS_COLUMNS,
        DROP_INSTANCES,
        NLP_EMOJI,
        DROP_INSTANCES_TRAIN,
        DROP_INSTANCES_DEV,
        _, # DROP_INSTANCES_TEST no hace falta porque aqui no hay test
        STOP_WORDS,
        TF_IDF_PICKLE_NAME,
        BIN_TFIDF
    ) = get_config(config)

    BINNING_ARG = None
    if BIN_TFIDF:
        BINNING_ARG = 10
    CONFIG = None
    ALGORITHMS = []
    for algorithm in algorithms:
        if algorithm["algorithm"] == Algorithm.KNN:
            try:
                knn_config = config["knn_config"]
            except:
                print("Si seleccionas \"KNN\" tienes que poner \"knn_config\"")
                exit(1)
            mink = knn_config["minK"]
            maxk = knn_config["maxK"]
            stepk = get_att_default(knn_config, "stepK", 1)
            minp = knn_config["minP"]
            maxp = knn_config["maxP"]
            weights = get_att_default(knn_config, "weights", ["uniform"])
            impute = json_bool(get_att_default(algorithm, "impute", "true"))
            drop = json_bool(get_att_default(algorithm, "drop", "true"))
            preprocess_categorical = json_bool(get_att_default(algorithm, "preprocess_categorical", "true"))
            preprocess_text = json_bool(get_att_default(algorithm, "preprocess_text", "true"))
            preprocess_numerical = json_bool(get_att_default(algorithm, "preprocess_numerical", "true"))
            rescale = json_bool(get_att_default(algorithm, "rescale", "true"))
            maximize_value = algorithm["maximize_output"]["property"]
            maximize_average = get_att_default(algorithm["maximize_output"], "average", None)
            name = algorithm["name"]
            ALGORITHMS.append(KnnConfig(
                mink,
                maxk,
                stepk,
                minp,
                maxp,
                weights,
                impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale,
                maximize_value,
                maximize_average, # type: ignore
                name
                ))
        elif algorithm["algorithm"] == Algorithm.DecisionTree:
            try:
                decision_tree_config = config["decision_tree_config"]
            except:
                print("Si seleccionas \"DecisionTree\" tienes que poner \"decision_tree_config\"")
                exit(1)
            impute = json_bool(get_att_default(algorithm, "impute", "true"))
            drop = json_bool(get_att_default(algorithm, "drop", "true"))
            preprocess_categorical = json_bool(get_att_default(algorithm, "preprocess_categorical", "true"))
            preprocess_text = json_bool(get_att_default(algorithm, "preprocess_text", "true"))
            preprocess_numerical = json_bool(get_att_default(algorithm, "preprocess_numerical", "true"))
            rescale = json_bool(get_att_default(algorithm, "rescale", "false"))
            criterios = decision_tree_config["criterios"]
            splitters = decision_tree_config["splitter"]
            min_maxDepth = decision_tree_config["min_maxDepth"]
            max_maxDepth = decision_tree_config["max_maxDepth"]
            min_minSamplesLeaf = decision_tree_config["min_minSamplesLeaf"]
            max_minSamplesLeaf = decision_tree_config["max_minSamplesLeaf"]
            maximize_value = algorithm["maximize_output"]["property"]
            maximize_average = get_att_default(algorithm["maximize_output"], "average", None)
            name = algorithm["name"]
            binning = json_bool(get_att_default(decision_tree_config, "binning", "true"))
            BINNING_ARG = decision_tree_config["numbins"]
            ALGORITHMS.append(DecisionTreeConfig(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale,
                criterios,
                splitters,
                min_maxDepth,
                max_maxDepth,
                min_minSamplesLeaf,
                max_minSamplesLeaf,
                maximize_value,
                maximize_average, # type: ignore
                name,
                binning))
        elif algorithm["algorithm"] == Algorithm.RandomForest:
            try:
                random_forest_config = config["random_forest_config"]
            except:
                print("Si seleccionas \"RandomForest\" tienes que poner \"random_forest_config\"")
                exit(1)
            try:
                decision_tree_config = config["decision_tree_config"]
            except:
                print("Si seleccionas \"RandomForest\" tienes que poner \"decision_tree_config\"")
                exit(1)
            impute = json_bool(get_att_default(algorithm, "impute", "true"))
            drop = json_bool(get_att_default(algorithm, "drop", "true"))
            preprocess_categorical = json_bool(get_att_default(algorithm, "preprocess_categorical", "true"))
            preprocess_text = json_bool(get_att_default(algorithm, "preprocess_text", "true"))
            preprocess_numerical = json_bool(get_att_default(algorithm, "preprocess_numerical", "true"))
            rescale = json_bool(get_att_default(algorithm, "rescale", "false"))
            maximize_value = algorithm["maximize_output"]["property"]
            maximize_average = get_att_default(algorithm["maximize_output"], "average", None)
            name = algorithm["name"]
            binning = json_bool(get_att_default(decision_tree_config, "binning", "true"))
            BINNING_ARG = decision_tree_config["numbins"]
            criterios = decision_tree_config["criterios"]
            splitters = decision_tree_config["splitter"]
            min_maxDepth = decision_tree_config["min_maxDepth"]
            max_maxDepth = decision_tree_config["max_maxDepth"]
            min_minSamplesLeaf = decision_tree_config["min_minSamplesLeaf"]
            max_minSamplesLeaf = decision_tree_config["max_minSamplesLeaf"]
            n_estimators = random_forest_config["n_estimators"]
            ALGORITHMS.append(RandomForestConfig(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale,
                maximize_value,
                maximize_average, # type: ignore
                criterios,
                splitters,
                min_maxDepth,
                max_maxDepth,
                min_minSamplesLeaf,
                max_minSamplesLeaf,
                name,
                binning,
                n_estimators))
        elif algorithm["algorithm"] == Algorithm.NaiveBayes:
            try:
                naive_bayes_config = config["naive_bayes_config"]
            except:
                print("Si seleccionas \"NaiveBayes\" tienes que poner \"naive_bayes_config\"")
                exit(1)
            impute = json_bool(get_att_default(algorithm, "impute", "true"))
            drop = json_bool(get_att_default(algorithm, "drop", "true"))
            preprocess_categorical = json_bool(get_att_default(algorithm, "preprocess_categorical", "true"))
            preprocess_text = json_bool(get_att_default(algorithm, "preprocess_text", "true"))
            preprocess_numerical = json_bool(get_att_default(algorithm, "preprocess_numerical", "true"))
            rescale = json_bool(get_att_default(algorithm, "rescale", "false"))
            maximize_value = algorithm["maximize_output"]["property"]
            maximize_average = get_att_default(algorithm["maximize_output"], "average", None)
            name = algorithm["name"]
            type_ = naive_bayes_config["type"]
            binning = json_bool(get_att_default(naive_bayes_config, "binning", "false"))

            min_alpha = get_att_default(naive_bayes_config, "min_alpha", 1.0)
            max_alpha = get_att_default(naive_bayes_config, "max_alpha", 1.01)
            step_alpha = get_att_default(naive_bayes_config, "step_alpha", 0.1)
            if binning:
                BINNING_ARG = naive_bayes_config["numbins"]
            ALGORITHMS.append(NaiveBayesConfig(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale,
                maximize_value,
                maximize_average, # type: ignore
                name,
                type_,
                binning,
                min_alpha,
                max_alpha,
                step_alpha))
        else:
            print(algorithm["algorithm"] == Algorithm.DecisionTree)
            print(algorithm["algorithm"])
            print(Algorithm.DecisionTree)
            print(f"No se reconoce el algoritmo {algorithm}")

    if len(ALGORITHMS) == 0:
        print("No has seleccionado ningún algoritmo")
        exit(1)

    infos = []
    for algorithm in ALGORITHMS:
        infos.append(run(
            ml_dataset,
            COLUMNS,
            PUNTO_A_COMA_COLUMNS,
            CATEGORICAL_COLUMNS,
            TEXT_COLUMNS,
            NUMERICAL_COLUMNS,
            PREDICT_COLUMN,
            TARGET_MAP,
            DROP_ROWS_WHEN_MISSING,
            IMPUTE_WHEN_MISSING,
            TEST_SIZE,
            RESCALE_FEATURES,
            UNDERSAMPLING_RATIO,
            OVERSAMPLING_RATIO,
            algorithm,
            BINNING_COLUMNS,
            BINNING_ARG,
            TF_IDF_COLUMNS,
            DROP_AFTER_PREPROCESS_COLUMNS,
            DROP_INSTANCES,
            NLP_EMOJI,
            DROP_INSTANCES_TRAIN,
            DROP_INSTANCES_DEV,
            STOP_WORDS,
            TF_IDF_PICKLE_NAME,
            BIN_TFIDF
        ))
    with open("datos_ultima_ejecucion.txt", "w") as f:
        for info in infos:
            f.write(info)
