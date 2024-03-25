from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import F, col, max as max_, abs as abs_, row_number, count, last

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
        # Add a unique identifier to join samples later
        data = data.withColumn("id", F.monotonically_increasing_id())
        
        # Prepare window specifications for cumulative distribution
        windowSpec1 = Window.orderBy(col(self.sample1_column)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        windowSpec2 = Window.orderBy(col(self.sample2_column)).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        
        # Calculate ranks
        data = data.withColumn("rank_sample1", F.percent_rank().over(windowSpec1))
        data = data.withColumn("rank_sample2", F.percent_rank().over(windowSpec2))
        
        # Prepare a combined column for calculating max difference
        data = data.withColumn("combined_rank", F.when(col(self.sample1_column).isNotNull(), col("rank_sample1")).otherwise(col("rank_sample2")))
        
        # Calculate KS statistic as the maximum absolute difference in ranks
        ks_statistic = data.select(max_(F.abs(col("rank_sample1") - col("rank_sample2"))).alias("ks_stat")).collect()[0]["ks_stat"]
        
        return ks_statistic