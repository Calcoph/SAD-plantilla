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

class Enum:
    pass

class ColumnType(Enum):
    CATEGORICAL = "categorical"
    TEXT = "text"
    NUMERICAL = "numerical"

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

class ImputeMethod(Enum):
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    CREATE_CATEGORY = "CREATE_CATEGORY"
    MODE = "MODE"
    CONSTANT = "CONSTANT"

class RescaleMethod(Enum):
    MINMAX = "MINMAX"
    AVGSTD = "AVGSTD"

class RescaleFeature:
    def __init__(self, feature: str, method: str):
        self.feature = feature
        self.method = method

class ImputeFeature:
    def __init__(self, feature: str, method: str, value=None):
        self.feature = feature
        self.method = method
        self.value = value
