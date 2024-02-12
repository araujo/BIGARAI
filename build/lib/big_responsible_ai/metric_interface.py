
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class Metric(ABC):
    """
    Abstract base class for responsible AI metrics.
    """

    @abstractmethod
    def compute(self, data: DataFrame) -> float:
        """
        Compute the metric on the given DataFrame.

        Parameters:
        data (DataFrame): The input Spark DataFrame to compute the metric on.

        Returns:
        float: The computed metric value.
        """
        pass
