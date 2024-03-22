from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, avg

class ConditionalDemographicDisparity:
    def __init__(self, outcome_column: str, demographic_column: str, control_column: str):
        """
        Initialize the ConditionalDemographicDisparity class with the names of the outcome, 
        demographic, and control columns.

        Parameters:
        - outcome_column (str): The name of the column for the outcome variable in the DataFrame.
        - demographic_column (str): The name of the column for the demographic variable in the DataFrame.
        - control_column (str): The name of the column for the control variable in the DataFrame.
        """
        self.outcome_column = outcome_column
        self.demographic_column = demographic_column
        self.control_column = control_column

    def compute(self, data: DataFrame) -> DataFrame:
        # Calculate the average outcome for each demographic group within each control group
        conditional_outcomes = data.groupBy(self.control_column, self.demographic_column) \
                                   .agg(avg(self.outcome_column).alias("avg_outcome"))

        # Pivot the DataFrame to have demographic groups as columns
        pivoted = conditional_outcomes.groupBy(self.control_column) \
                                      .pivot(self.demographic_column) \
                                      .avg("avg_outcome")

        # Ensure there are at least two demographic groups to calculate disparity
        demographic_groups = pivoted.columns
        demographic_groups.remove(self.control_column)
        if len(demographic_groups) < 2:
            raise ValueError("Not enough demographic groups to calculate disparity.")

        # Calculate disparities between all pairs of demographic groups
        # Here, we simply subtract the values of the first two demographic groups as an example
        disparity = pivoted.withColumn("disparity", col(demographic_groups[1]) - col(demographic_groups[0]))

        return disparity
