from pyspark.sql import DataFrame
from pyspark.sql.functions import col, log2

class KLDivergence(Metric):
    def __init__(self, actual_column: str, predicted_column: str):
        self.actual_column = actual_column
        self.predicted_column = predicted_column

    def compute(self, data: DataFrame) -> float:
        # Add a small number to avoid log(0)
        epsilon = 1e-9

        # Calculate KL Divergence using DataFrame API
        kl_divergence = data \
            .select(
                col(self.actual_column),
                col(self.predicted_column)
            ) \
            .withColumn("kl_term", 
                        col(self.actual_column) * log2((col(self.actual_column) + epsilon) / (col(self.predicted_column) + epsilon))
            ) \
            .groupBy() \
            .sum("kl_term") \
            .collect()[0][0]

        return kl_divergence
