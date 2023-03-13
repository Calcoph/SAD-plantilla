import pandas as pd
import numpy as np
import sys
import json
import jsonschema
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from typing import TypeVar, cast

class Enum:
    pass

class ImputeMethod(Enum):
    MEAN = 0
    MEDIAN = 1
    CREATE_CATEGORY = 2
    MODE = 3
    CONSTANT = 4

class ImputeFeature:
    def __init__(self, feature: str, method: int, value=None):
        self.feature = feature
        self.method = method
        self.value = value

class RescaleMethod(Enum):
    MINMAX = 0
    AVGSTD = 1

class Algorithm(Enum):
    KNN = "KNN"
    DecisionTree = "DecisionTree",
    RandomForest = "RandomForest",

class RescaleFeature:
    def __init__(self, feature: str, method: int):
        self.feature = feature
        self.method = method

class AlgorithmConfig:
    def __init__(self,
                impute: bool,
                drop: bool,
                preprocess_categorical: bool,
                preprocess_text: bool,
                preprocess_numerical: bool,
                rescale: bool
        ) -> None:
        self.impute = impute
        self.drop = drop
        self.preprocess_categorical = preprocess_categorical
        self.preprocess_text = preprocess_text
        self.preprocess_numerical = preprocess_numerical
        self.rescale = rescale

class KnnConfig(AlgorithmConfig):
    def __init__(self,
                mink: int,
                maxk: int,
                stepk: int,
                minp: int,
                maxp: int,
                weights: str,
                impute: bool,
                drop: bool,
                preprocess_categorical: bool,
                preprocess_text: bool,
                preprocess_numerical: bool,
                rescale: bool
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
                rescale
            )

class DecisionTreeConfig(AlgorithmConfig):
    def __init__(self,
                impute: bool,
                drop: bool,
                preprocess_categorical: bool,
                preprocess_text: bool,
                preprocess_numerical: bool,
                rescale: bool
            ):
        super().__init__(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale)

class RandomForestConfig(AlgorithmConfig):
    def __init__(self,
                impute: bool,
                drop: bool,
                preprocess_categorical: bool,
                preprocess_text: bool,
                preprocess_numerical: bool,
                rescale: bool
            ):
        super().__init__(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale)

def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)


TARGET = "__target__"
def run(
        input_file,
        columns,
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
        algorithm: AlgorithmConfig
    ):

    ml_dataset = pd.read_csv(input_file)
    ml_dataset = ml_dataset[columns]

    for feature in categorical_features:
        # transforma las categóricas a unicode
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        # transforma las de texto a unicode
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    if algorithm.preprocess_numerical:
        for feature in numerical_features:
            # M8[ns] = nanosegundos desde 1 jan 1970
            if (ml_dataset[feature].dtype == np.dtype("M8[ns]")\
                or (\
                    hasattr(ml_dataset[feature].dtype, "base")\
                    and ml_dataset[feature].dtype.base == np.dtype("M8[ns]")\
                )):

                # si es fecha, convierte a fecha normal
                #ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
                print("Hay fechas")
                exit(1)
            else:
                # si no, lo convierte a double
                ml_dataset[feature] = ml_dataset[feature].astype('double')
    else:
        for feature in numerical_features:
            ml_dataset[feature] = ml_dataset[feature].astype('double')

    #convierte esta columna en un int de python, independientemente de su tipo original y lo mete en una nueva columna
    ml_dataset[TARGET] = ml_dataset[predict_column].map(str).map(target_map)
    del ml_dataset[predict_column]

    ml_dataset = ml_dataset[~ml_dataset[TARGET].isnull()]

    (train, test) = train_test_split(ml_dataset,test_size=test_size,random_state=42,stratify=ml_dataset[[TARGET]])

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
    
    trainX = train.drop(TARGET, axis=1) # type: ignore
    #trainY = train[TARGET]

    testX = test.drop(TARGET, axis=1) # type: ignore
    #testY = test[TARGET]

    trainY = np.array(train[TARGET]) # type: ignore
    testY = np.array(test[TARGET]) # type: ignore

    # Explica lo que se hace en este paso
    # TODO: Solo undersamplear/oversamplear si están desbalanceados
    if undersampling_ratio is not None:
        undersample = RandomUnderSampler(sampling_strategy=undersampling_ratio)#la mayoria va a estar representada el doble de veces

        trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY) # type: ignore
        testXUnder,testYUnder = undersample.fit_resample(testX, testY) # type: ignore

        trainX,trainY = trainXUnder, trainYUnder
        testX,testY = testXUnder, testYUnder

    # Explica lo que se hace en este paso
    print(len(target_map))
    if isinstance(algorithm, KnnConfig):
        algorithm = cast(KnnConfig, algorithm)
        run_knn(algorithm, trainX, trainY, testX, testY, target_map, test)
    if isinstance(algorithm, RandomForestConfig):
        algorithm = cast(RandomForestConfig, algorithm)
        run_random_forest(algorithm)
    if isinstance(algorithm, DecisionTreeConfig):
        algorithm = cast(DecisionTreeConfig, algorithm)
        run_decision_tree(algorithm)
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
    # TODO: Escribir en un archivo que k, p se han utilizado. También todos los parámetros del algoritmo
    # TODO: Escribir en ese archivo también más medidas
    for k in range(algorithm.mink, algorithm.maxk+1, algorithm.stepk):
        for p in range(algorithm.minp, algorithm.maxp):
            clf = KNeighborsClassifier(
                n_neighbors=k,
                weights=algorithm.weights,
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

            evaluate(predictions, probas, testX, testY, target_map, test)

def run_random_forest(
    algorithm: RandomForestConfig
):
    pass

def run_decision_tree(
    algorithm: DecisionTreeConfig
):
    pass

def evaluate(predictions, probas, testX, testY, target_map, test):
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

    print("fscore:")
    print(f1_score(testY, predictions, average=None)) # type: ignore
    print()
    print("classification report:")
    print(classification_report(testY,predictions))
    print()
    print("confusion matrix:")
    print(confusion_matrix(testY, predictions, labels=[1,0]))
    print("fin")

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

if __name__ == "__main__":
    if sys.version_info < (3, 0):
        print("Estás usando python 2, usa el 3")
        exit(1)

    with open("config_iris.json") as f:
        config = json.load(f)
        with open(config["$schema"]) as f:
            schema = json.load(f)
        try:
            jsonschema.validate(config, schema)
        except Exception as e:
            print(e)

    INPUT_FILE = config["input_file"]
    COLUMNS = config["columns"]
    CATEGORICAL_COLUMNS = config["categorical_columns"]
    for column in CATEGORICAL_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de categorical_columns no está en columns")
    TEXT_COLUMNS = config["text_columns"]
    for column in CATEGORICAL_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de text_columns no está en columns")
    NUMERICAL_COLUMNS = config["numerical_columns"]
    for column in CATEGORICAL_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de numerical_columns no está en columns")
    PREDICT_COLUMN = config["predict_column"]
    TARGET_MAP = config["target_map"]
    DROP_ROWS_WHEN_MISSING = config["drop_rows_when_missing"]
    for column in CATEGORICAL_COLUMNS:
        if column not in COLUMNS:
            print(f"{column} de drop_rows_when_missing no está en columns")
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
        RESCALE_FEATURES.append(RescaleFeature(column, method))
    try:
        UNDERSAMPLING_RATIO = config["undersampling_ratio"]
    except KeyError:
        UNDERSAMPLING_RATIO = None
    
    algorithms = config["algorithms"]
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
            weights = get_att_default(knn_config, "weights", "uniform")
            impute = json_bool(get_att_default(algorithm, "impute", "true"))
            drop = json_bool(get_att_default(algorithm, "drop", "true"))
            preprocess_categorical = json_bool(get_att_default(algorithm, "preprocess_categorical", "true"))
            preprocess_text = json_bool(get_att_default(algorithm, "preprocess_text", "true"))
            preprocess_numerical = json_bool(get_att_default(algorithm, "preprocess_numerical", "true"))
            rescale = json_bool(get_att_default(algorithm, "rescale", "true"))
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
                rescale
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
            ALGORITHMS.append(DecisionTreeConfig(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale))
        elif algorithm["algorithm"] == Algorithm.RandomForest:
            try:
                random_forest_config = config["random_forest_config"]
            except:
                print("Si seleccionas \"RandomForest\" tienes que poner \"random_forest_config\"")
                exit(1)
            impute = json_bool(get_att_default(algorithm, "impute", "true"))
            drop = json_bool(get_att_default(algorithm, "drop", "true"))
            preprocess_categorical = json_bool(get_att_default(algorithm, "preprocess_categorical", "true"))
            preprocess_text = json_bool(get_att_default(algorithm, "preprocess_text", "true"))
            preprocess_numerical = json_bool(get_att_default(algorithm, "preprocess_numerical", "true"))
            rescale = json_bool(get_att_default(algorithm, "rescale", "false"))
            ALGORITHMS.append(DecisionTreeConfig(impute,
                drop,
                preprocess_categorical,
                preprocess_text,
                preprocess_numerical,
                rescale))
        else:
            print(f"No se reconoce el algoritmo {algorithm}")

    if len(ALGORITHMS) == 0:
        print("No has seleccionado ningún algoritmo")
        exit(1)
    # TODO: al igual que en el knn se cambia el k, en los otros se puede cambiar como se hace el binning.
    for algorithm in ALGORITHMS:
        print(type(algorithm))
        run(
            INPUT_FILE,
            COLUMNS,
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
            algorithm
        )
