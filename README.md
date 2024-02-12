
# Responsible AI Metrics

This package provides a set of metrics implemented in PySpark for assessing responsible AI practices. It includes metrics such as Class Imbalance, Difference in Proportion of Labels, Kullback-Leibler Divergence, Lp-Norm, Total Variation Distance, Kolmogorov-Smirnov, and Conditional Demographic Disparity. These metrics can be used to evaluate and ensure fairness, equity, and ethical use of AI technologies.


## Installation

To install this package, make sure you have PySpark installed in your environment. Then, you can install this package using pip:

```bash
pip install bigrai.
```
Alternatively, if you have cloned this package from a repository, navigate to the root directory of the package and run:

```bash
pip install .
```


## Usage

To use the metrics in this package, first import the desired metric class from the package and then initialize it with the appropriate parameters. After that, you can compute the metric by passing a PySpark DataFrame to the compute method of the metric class.

```
from pyspark.sql import SparkSession
from responsible_ai.class_imbalance import ClassImbalance

# Initialize a Spark session
spark = SparkSession.builder.appName("ResponsibleAIMetrics").getOrCreate()

# Sample data
data = spark.createDataFrame(
    [(0,), (1,), (1,), (1,)], 
    ["label"]
)

# Initialize the ClassImbalance metric
class_imbalance = ClassImbalance(label_column="label")

# Compute the class imbalance
imbalance_ratio = class_imbalance.compute(data)
print("Class Imbalance Ratio:", imbalance_ratio)

```
#Other Metrics

You can use other metrics in a similar fashion by importing the corresponding class and initializing it with relevant parameters. Here's a generic template:

```
from responsible_ai.<metric_module> import <MetricClass>

# Initialize the metric
metric = <MetricClass>(param1="column1", param2="column2", ...)

# Compute the metric
result = metric.compute(data)

```

Replace <metric_module>, <MetricClass>, and parameters (param1, param2, ...) with the actual module name, class name, and parameters of the metric you want to use.

#Contributing

We welcome contributions to improve this package. If you have suggestions or improvements, please submit a pull request or open an issue in the repository.

#License

This project is freely available for use, modification, and distribution under the terms of the MIT License. The MIT License is a permissive free software license that puts only very limited restriction on reuse and therefore it allows for maximum flexibility in the use of the software.

By using this project, you agree to abide by the terms of the MIT License. A copy of the license is included in this project's repository in the LICENSE file. For more details, please see MIT License.

Remember to include a LICENSE file in the root of your project's repository containing the full text of the MIT License. The license text can be found on the Open Source Initiative's website or other reputable sources. Including this file is a standard practice for open-source projects and is a requirement for most open-source licenses, including the MIT License.

This approach ensures that your project's users are informed about their rights and obligations when using your software, in accordance with PyPI and open-source community standards.