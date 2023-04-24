# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import json

import modelo_proyecto
from modelo_proyecto import ImputeMethod, ImputeFeature, get_att_default, ColumnType, RescaleFeature, RescaleMethod, coerce_to_unicode, Algorithm

import jsonschema

model=""
p="./"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'p:m:c:h',['path=','model=', 'config=','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-p','--path'):
            p = arg
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('-c', '--config'):
            c = arg
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName\n -c config.json')
            exit(1)

    if sys.version_info < (3, 0):
        print("Estás usando python 2, usa el 3")
        exit(1)

    with open(c) as f: # type: ignore
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
        _,# DROP_INSTANCES_TRAIN no hace falta porque aqui no hay test
        _,# DROP_INSTANCES_DEV no hace falta porque aqui no hay test
        DROP_INSTANCES_TEST, # DROP_INSTANCES_TEST no hace falta porque aqui no hay test
        STOP_WORDS,
        TF_IDF_PICKLE_NAME
    ) = modelo_proyecto.get_config(config)

    if p == './':
        model=p+str(m) # type: ignore
    else:
        model=p+"/"+str(m) # type: ignore
        

    #Abrir el fichero .csv con las instancias a predecir y que no contienen la clase y cargarlo en un dataframe de pandas para hacer la prediccion
    y_test=pd.DataFrame()
    algorithm = None
    for alg in config["algorithms"]:
        if alg["name"] == m: # type: ignore
            algorithm = alg
            break
    
    if algorithm is None:
        print("No se ha encontrado el algoritmo")
        exit(1)

    if algorithm["algorithm"] == Algorithm.DecisionTree:
        binning = modelo_proyecto.json_bool(get_att_default(config["decision_tree_config"], "binning", "true"))
    elif algorithm["algorithm"] == Algorithm.RandomForest:
        binning = modelo_proyecto.json_bool(get_att_default(config["decision_tree_config"], "binning", "true"))
    else:
        binning = False

    try:
        with open("last_binnings.json") as f:
            binning_arg = json.load(f)
    except:
        binning_arg = None

    testX = modelo_proyecto.preprocess(
        ml_dataset,
        COLUMNS,
        PUNTO_A_COMA_COLUMNS,
        CATEGORICAL_COLUMNS,
        TEXT_COLUMNS,
        NUMERICAL_COLUMNS,
        PREDICT_COLUMN,
        TARGET_MAP,
        modelo_proyecto.json_bool(get_att_default(algorithm, "preprocess_numerical", "true")),
        binning,
        BINNING_COLUMNS,
        binning_arg,
        TF_IDF_COLUMNS,
        DROP_INSTANCES,
        NLP_EMOJI,
        STOP_WORDS,
        True,
        TF_IDF_PICKLE_NAME
    )

    if algorithm["algorithm"] == Algorithm.KNN:
        drop = modelo_proyecto.json_bool(get_att_default(algorithm, "drop", "true"))
        impute = modelo_proyecto.json_bool(get_att_default(algorithm, "drop", "true"))
        rescale = modelo_proyecto.json_bool(get_att_default(algorithm, "rescale", "true"))
    elif algorithm["algorithm"] == Algorithm.DecisionTree:
        drop = modelo_proyecto.json_bool(get_att_default(algorithm, "drop", "true"))
        impute = modelo_proyecto.json_bool(get_att_default(algorithm, "drop", "true"))
        rescale = modelo_proyecto.json_bool(get_att_default(algorithm, "rescale", "false"))
    elif algorithm["algorithm"] == Algorithm.RandomForest:
        drop = modelo_proyecto.json_bool(get_att_default(algorithm, "drop", "true"))
        impute = modelo_proyecto.json_bool(get_att_default(algorithm, "drop", "true"))
        rescale = modelo_proyecto.json_bool(get_att_default(algorithm, "rescale", "false"))
    else:
        print(f"No se reconoce el algoritmo {algorithm}")
        exit(1)

    if drop:
        # Explica lo que se hace en este paso
        for feature in DROP_ROWS_WHEN_MISSING:
            # se queda solo con las instancias que no tienen valores nullos
            testX = testX[testX[feature].notnull()]
            print('Dropped missing records in %s' % feature)

    if impute:
        # Explica lo que se hace en este paso
        for feature in IMPUTE_WHEN_MISSING:
            # Calcula el valor para reemplazar los na
            v = None;
            if feature.method == ImputeMethod.MEAN:
                v = testX[feature.feature].mean() # type: ignore
            elif feature.method == ImputeMethod.MEDIAN:
                v = testX[feature.feature].median() # type: ignore
            elif feature.method == ImputeMethod.CREATE_CATEGORY:
                v = 'NULL_CATEGORY'
            elif feature.method == ImputeMethod.MODE:
                v = testX[feature.feature].value_counts().index[0] # type: ignore
            elif feature.method == ImputeMethod.CONSTANT:
                v = feature.value

            if v is None:
                print("No hay v")
                exit(1)

            # Llena los na con el valor calculado
            testX[feature.feature] = testX[feature.feature].fillna(v) # type: ignore
            print('Imputed missing values in feature %s with value %s' % (feature.feature, coerce_to_unicode(v)))

    if rescale:
        for feature in RESCALE_FEATURES:
            if feature.method == RescaleMethod.MINMAX:
                _min = testX[feature.feature].min() # type: ignore
                _max = testX[feature.feature].max() # type: ignore
                scale = _max - _min
                shift = _min
            elif feature.method == RescaleMethod.AVGSTD:
                shift = testX[feature.feature].mean() # type: ignore
                scale = testX[feature.feature].std() # type: ignore
            else:
                print("Bad rescale method")
                exit(1)
            if scale == 0.:
                del testX[feature.feature] # type: ignore
                print('Feature %s was dropped because it has no variance' % feature.feature)
            else:
                print('Rescaled %s' % feature.feature)
                testX[feature.feature] = (testX[feature.feature] - shift).astype(np.float64) / scale # type: ignore

    del testX["__target__"]

    for column in DROP_AFTER_PREPROCESS_COLUMNS:
        testX.drop(column, axis=1, inplace=True)

    for drop_instance in DROP_INSTANCES_TEST:
        if drop_instance.comparison == modelo_proyecto.DropInstance.EQUAL:
            testX.drop(testX[testX[drop_instance.column] == drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == modelo_proyecto.DropInstance.LESS:
            testX.drop(testX[testX[drop_instance.column] < drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == modelo_proyecto.DropInstance.LESS_EQ:
            testX.drop(testX[testX[drop_instance.column] <= drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == modelo_proyecto.DropInstance.MORE:
            testX.drop(testX[testX[drop_instance.column] > drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == modelo_proyecto.DropInstance.MORE_EQ:
            testX.drop(testX[testX[drop_instance.column] >= drop_instance.value].index, inplace = True)
        elif drop_instance.comparison == modelo_proyecto.DropInstance.UNEQUAL:
            testX.drop(testX[testX[drop_instance.column] != drop_instance.value].index, inplace = True)
        else:
            print("Error inesperado")
            exit(1)

    print(testX.head(5))
    clf = pickle.load(open(model, 'rb'))
    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    y_test['preds'] = predictions
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    results_test = testX.join(predictions, how='left')
    
    print(results_test)

    print("valores de las predicciones:")
    target_map = config["target_map"]
    for key, value in target_map.items():
        print(f"    {value} es {key}")
