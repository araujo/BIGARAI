from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pow, abs, sum as sum_
from metric_interface import Metric

class LpNorm(Metric):
    def __init__(self, p: float, column1: str, column2: str):
        """
        Initialize the LpNorm with the power p and the names of the two columns to compare.

        Parameters:
        - p (float): The exponent value in the Lp-Norm.
        - column1 (str): The name of the first column in the DataFrame.
        - column2 (str): The name of the second column in the DataFrame.
        """
        self.p = p
        self.column1 = column1
        self.column2 = column2

    def compute(self, data: DataFrame) -> float:
        # Calculate the Lp-Norm
        lp_norm = data.select(
            pow(abs(col(self.column1) - col(self.column2)), self.p)
        ).agg(
            pow(sum_("pow"), 1 / self.p)
        ).collect()[0][0]

        return lp_norm

def test_lp_norm(spark):
    """
    Test for the LpNorm metric.
    """
    # Sample data for two sequences to compare
    data = spark.createDataFrame([
        (1, 3),
        (2, 4),
        (3, 5)
    ], ["column1", "column2"])

    # Initialize the LpNorm with exponent p and column names
    p_value = 2  # Using L2 norm as an example
    metric = LpNorm(p=p_value, column1="column1", column2="column2")

    # Compute the Lp-Norm
    lp_norm_value = metric.compute(data)

    # Expected Lp-Norm value (you'll need to compute this based on your sample data or use a known value for a test case)
    expected_lp_norm = ...  # This value should be calculated based on your test data

    # Assert that the computed Lp-Norm is as expected
    assert lp_norm_value == pytest.approx(expected_lp_norm, abs=1e-2)
