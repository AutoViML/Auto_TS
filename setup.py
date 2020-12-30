import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_ts",
    version="0.0.25",
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
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn==0.22.2",
        "fbprophet",
        "statsmodels",
        "tscv"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
