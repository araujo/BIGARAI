from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, count, rank, max as max_, abs as abs_
from pyspark.sql.window import Window



class KolmogorovSmirnov:
    def __init__(self, data, filter_column, filter_value1, filter_value2, sample_column):
        self.data = data
        self.filter_column = filter_column
        self.filter_value1 = filter_value1
        self.filter_value2 = filter_value2
        self.sample_column = sample_column

    def compute(self):
        # Filter the data for the two samples
        sample1 = self.data.filter(col(self.filter_column) == self.filter_value1).select(self.sample_column)
        sample2 = self.data.filter(col(self.filter_column) == self.filter_value2).select(self.sample_column)

        # Combine samples with a marker to distinguish them
        sample1 = sample1.withColumn("sample", lit(1))
        sample2 = sample2.withColumn("sample", lit(2))

        # Combine the two samples into a single DataFrame
        combined = sample1.unionByName(sample2)

        # Calculate the cumulative distribution for each value in each sample
        windowSpec = Window.orderBy(self.sample_column).rowsBetween(Window.unboundedPreceding, Window.currentRow)
        combined = combined.withColumn('cum_dist', count(self.sample_column).over(windowSpec) / count(self.sample_column).over(Window.partitionBy("sample")))

        # Calculate the KS statistic as the maximum difference in cumulative distribution between the two samples
        ks_statistic = combined.groupBy("sample").agg(max_("cum_dist").alias("max_cum_dist")).agg(max_("max_cum_dist") - min_("max_cum_dist")).collect()[0][0]

        return ks_statistic

