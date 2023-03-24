import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_ts",
    version="0.0.69",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Automatically Build Multiple Time Series models fast - now with Facebook Prophet!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/Auto_TS",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "ipython",
        "jupyter",
        "pmdarima",
        "numpy",
        "xlrd",
        "pandas",
        "matplotlib",
        "seaborn",
        "prophet",
        "scikit-learn>=0.24.0",
        "statsmodels",
        "xgboost>=1.5.1",
        "prettytable",
        "dask>=2022.1.0",
        "pyyaml==5.4.1",
        "distributed>=2022.2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
