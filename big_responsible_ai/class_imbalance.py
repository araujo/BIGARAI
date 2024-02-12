from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, max as max_, min as min_

class ClassImbalance(Metric):
    def __init__(self, label_column: str):
        """
        Initialize the ClassImbalance class with the name of the label column.

        Parameters:
        - label_column (str): The name of the column containing class labels in the DataFrame.
        """
        self.label_column = label_column

    def compute(self, data: DataFrame) -> float:
        # Group the data by the label column and count the number of instances in each class
        class_counts = data.groupBy(self.label_column).agg(count("*").alias("count"))

        # Find the maximum and minimum counts among the classes
        max_count = class_counts.agg(max_("count").alias("max_count")).collect()[0]["max_count"]
        min_count = class_counts.agg(min_("count").alias("min_count")).collect()[0]["min_count"]

        # Calculate the class imbalance ratio as the ratio of the maximum count to the minimum count
        # Adding a small value to the denominator to avoid division by zero
        imbalance_ratio = max_count / (min_count + 1e-9)

        return imbalance_ratio
