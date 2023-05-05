import pandas as pd
from typing import TypeVar

from clases import ColumnType, DropInstance, ImputeFeature, ImputeMethod, NlpEmoji, RescaleFeature, RescaleMethod, StopWords

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
    
    delete_emoji = get_att_default(config, "delete_emoji", [])
    delete_emoji = []
    for item in delete_emoji:
        column = item["column"]
        if column not in COLUMNS:
            print(f"{column} de delete_emoji no está en columns")
            exit(1)
        delete = json_bool(item["delete_emojis"])
        metrics_list = item["metrics"]
        metrics = set() # Para asegurarse de que no hay repeticiones
        for metric in metrics_list:
            metrics.add(metric)

        bin = json_bool(item["bin"])
        delete_emoji.append(NlpEmoji(column, delete, metrics, bin))

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

    try:
        UNDERSAMPLING_RATIO = config["undersampling_ratio"]
    except KeyError:
        UNDERSAMPLING_RATIO = None

    try:
        OVERSAMPLING_RATIO = config["oversampling_ratio"]
    except KeyError:
        OVERSAMPLING_RATIO = None
    
    algorithm = config["algorithm"]

    TF_IDF_PICKLE_NAME = get_att_default(config, "tf_idf_pickle_name", "default_tf_save.tfidf")
    BIN_TFIDF = json_bool(get_att_default(config, "bin_tfidf", "false"))

    return (
        algorithm,
        ml_dataset,
        COLUMNS,
        PUNTO_A_COMA_COLUMNS,
        CATEGORICAL_COLUMNS,
        TEXT_COLUMNS,
        NUMERICAL_COLUMNS,
        DROP_ROWS_WHEN_MISSING,
        IMPUTE_WHEN_MISSING,
        RESCALE_FEATURES,
        UNDERSAMPLING_RATIO,
        OVERSAMPLING_RATIO,
        BINNING_COLUMNS,
        TF_IDF_COLUMNS,
        drop_instances,
        delete_emoji,
        stop_words,
        TF_IDF_PICKLE_NAME,
        BIN_TFIDF
    )