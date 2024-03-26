from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, max as max_, abs as abs_, monotonically_increasing_id, rank, broadcast
from pyspark.sql.window import Window

class KolmogorovSmirnov:
    def __init__(self, data: DataFrame, filter_column: str, filter_value1: str, filter_value2: str, sample_column: str):
        self.data = data
        self.filter_column = filter_column
        self.filter_value1 = filter_value1
        self.filter_value2 = filter_value2
        self.sample_column = sample_column

    def calculate_ecdf(self, data, ecdf_column_name):
        windowSpec = Window.orderBy(self.sample_column).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        data = data.withColumn("id", monotonically_increasing_id())
        data = data.withColumn(ecdf_column_name, rank().over(windowSpec) / lit(data.count()))
        return data.select("id", ecdf_column_name)

    def compute(self):
        sample1_data = self.data.filter(col(self.filter_column) == self.filter_value1).select(self.sample_column)
        sample2_data = self.data.filter(col(self.filter_column) == self.filter_value2).select(self.sample_column)

        sample1_ecdf = self.calculate_ecdf(sample1_data, "ecdf1")
        sample2_ecdf = self.calculate_ecdf(sample2_data, "ecdf2")

        # Join the ECDFs. Use broadcast for efficiency if one of the datasets is much smaller.
        ecdf_comparison = sample1_ecdf.join(broadcast(sample2_ecdf), "id")

        # Calculate the KS statistic as the maximum absolute difference between the two ECDFs
        ks_statistic = ecdf_comparison.select(max_(abs_(col("ecdf1") - col("ecdf2"))).alias("ks")).collect()[0]["ks"]

        return ks_statistic

