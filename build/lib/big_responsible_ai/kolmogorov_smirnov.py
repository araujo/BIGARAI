from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, max as max_, abs as abs_, row_number, count, last

class KolmogorovSmirnov(Metric):
    def __init__(self, sample1_column: str, sample2_column: str):
        """
        Initialize the KolmogorovSmirnov class with the names of the columns representing the two samples.

        Parameters:
        - sample1_column (str): The name of the column for the first sample in the DataFrame.
        - sample2_column (str): The name of the column for the second sample in the DataFrame.
        """
        self.sample1_column = sample1_column
        self.sample2_column = sample2_column

    def compute(self, data: DataFrame) -> float:
        # Calculate the ECDF for each sample
        windowSpec1 = Window.orderBy(col(self.sample1_column)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ecdf1 = data.withColumn('ecdf1', (row_number().over(windowSpec1) - 1) / count().over(Window.partitionBy(self.sample1_column)))

        windowSpec2 = Window.orderBy(col(self.sample2_column)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ecdf2 = data.withColumn('ecdf2', (row_number().over(windowSpec2) - 1) / count().over(Window.partitionBy(self.sample2_column)))

        # Combine the two ECDFs into one DataFrame
        combined = ecdf1.select(col(self.sample1_column).alias("value"), "ecdf1").union(
            ecdf2.select(col(self.sample2_column).alias("value"), "ecdf2")
        ).distinct().orderBy("value")

        # Fill forward the last known values for each ECDF (since some values might be present in one sample and not the other)
        filled = combined.withColumn("ecdf1", last("ecdf1", True).over(Window.orderBy("value").rowsBetween(Window.unboundedPreceding, 0))) \
                         .withColumn("ecdf2", last("ecdf2", True).over(Window.orderBy("value").rowsBetween(Window.unboundedPreceding, 0)))

        # Calculate the KS statistic as the maximum absolute difference between the two ECDFs
        ks_statistic = filled.withColumn("ks", abs_(col("ecdf1") - col("ecdf2"))).agg(max_("ks")).collect()[0][0]

        return ks_statistic
