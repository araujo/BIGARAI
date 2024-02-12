from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, lit

class DifferenceInProportion(Metric):
    def __init__(self, label_column: str):
        """
        Initialize the DifferenceInProportion class with the name of the label column.

        Parameters:
        - label_column (str): The name of the column containing class labels in the DataFrame.
        """
        self.label_column = label_column

    def compute(self, data: DataFrame) -> float:
        # Calculate the total number of instances
        total_count = data.count()

        # Calculate the proportion of each class
        proportions = data.groupBy(self.label_column).agg((count(self.label_column) / lit(total_count)).alias("proportion"))

        # Find the maximum and minimum proportions
        max_proportion = proportions.agg({"proportion": "max"}).collect()[0][0]
        min_proportion = proportions.agg({"proportion": "min"}).collect()[0][0]

        # Calculate the difference in proportions
        difference = max_proportion - min_proportion

        return difference
