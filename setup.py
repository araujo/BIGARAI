
from setuptools import setup, find_packages

setup(
    name='big_responsible_ai',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pyspark',
    ],
    tests_require=[
        'pytest',
    ],
    description='Responsible AI metrics implemented in PySpark.',
)
