
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

from big_responsible_ai.class_imbalance import ClassImbalance
from big_responsible_ai.difference_in_proportion_labels import DifferenceInProportion
from big_responsible_ai.conditional_demographic_disparity import ConditionalDemographicDisparity
from big_responsible_ai.kolmogorov_smirnov import KolmogorovSmirnov
from big_responsible_ai.kullback_leibler_divergence import KLDivergence
from big_responsible_ai.lp_norm import LpNorm



@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[2]").appName("MetricTests").getOrCreate()

def test_class_imbalance(spark):
    data = spark.createDataFrame([(0,), (1,), (1,)], ["label"])
    metric = ClassImbalance(label_column="label")
    assert metric.compute(data) == 1.0

def test_difference_in_proportion_labes(spark):
    data = spark.createDataFrame([(0,), (1,), (1,), (1,)], ["label"])
    metric = DifferenceInProportion(label_column="label")
    assert metric.compute(data) == 0.5

def test_conditional_demographic_disparity(spark):
    """
    Test for the ConditionalDemographicDisparity metric.
    """
    # Sample data
    data = spark.createDataFrame([
        (1, "A", "Control", 0.5),
        (2, "B", "Control", 0.7),
        (3, "A", "Treatment", 0.6),
        (4, "B", "Treatment", 0.8),
        (5, "A", "Control", 0.55),
        (6, "B", "Control", 0.75),
        (7, "A", "Treatment", 0.65),
        (8, "B", "Treatment", 0.85),
    ], ["id", "demographic", "control", "outcome"])

    # Initialize the ConditionalDemographicDisparity with column names
    metric = ConditionalDemographicDisparity(outcome_column="outcome", demographic_column="demographic", control_column="control")

    # Compute the metric
    result = metric.compute(data)

    # Expected result DataFrame (you might need to adjust this based on your actual expected results)
    expected_data = [
        ("Control", 0.55, 0.75, 0.20),  # Disparity in Control group
        ("Treatment", 0.625, 0.825, 0.20)  # Disparity in Treatment group
    ]
    expected_df = spark.createDataFrame(expected_data, ["control", "A", "B", "disparity"])

    # Compare the result with the expected DataFrame
    assert result.collect() == expected_df.collect()

def test_kolmogorov_smirnov(spark):
    """
    Test for the KolmogorovSmirnov metric.
    """
    # Sample data for two samples
    data = spark.createDataFrame([
        (0.1, 0.2),
        (0.4, 0.5),
        (0.3, 0.3),
        (0.6, 0.7),
        (0.8, 0.9)
    ], ["sample1", "sample2"])

    # Initialize the KolmogorovSmirnov with column names
    metric = KolmogorovSmirnov(sample1_column="sample1", sample2_column="sample2")

    # Compute the KS statistic
    ks_statistic = metric.compute(data)

    # Expected KS statistic (you'll need to compute this based on your sample data or use a known value for a test case)
    expected_ks_statistic = ...  # This value should be pre-calculated based on your test data

    # Assert that the computed KS statistic is as expected
    assert ks_statistic == pytest.approx(expected_ks_statistic, abs=1e-2)

def test_kl_divergence(spark):
    """
    Test for the KLDivergence metric.
    """
    # Sample data for actual and predicted distributions
    data = spark.createDataFrame([
        (0.2, 0.1),
        (0.4, 0.3),
        (0.4, 0.6)
    ], ["actual", "predicted"])

    # Initialize the KLDivergence with column names
    metric = KLDivergence(actual_column="actual", predicted_column="predicted")

    # Compute the KL Divergence
    kl_divergence = metric.compute(data)

    # Expected KL Divergence (you'll need to compute this based on your sample data or use a known value for a test case)
    expected_kl_divergence = ...  # This value should be calculated based on your test data

    # Assert that the computed KL Divergence is as expected
    assert kl_divergence == pytest.approx(expected_kl_divergence, abs=1e-2)

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
