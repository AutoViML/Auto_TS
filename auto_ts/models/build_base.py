from typing import Optional, List
from abc import ABC, abstractmethod
import pandas as pd # type: ignore
from pandas.core.generic import NDFrame # type:ignore

class BuildBase(ABC):
    """
    Base Class for Building a model
    """

    def __init__(self, scoring: str, forecast_period: int, verbose: int):
        self.scoring = scoring
        self.forecast_period = forecast_period
        self.verbose = verbose
        self.model = None
        self.original_target_col: str = ""
        self.original_preds: List[str] = []


    @abstractmethod
    def fit(self, ts_df: pd.DataFrame, target_col: str, cv: Optional[int] = None) -> object:
        """
        Fits the model to the data

        :param ts_df The time series data to be used for fitting the model
        :type ts_df pd.DataFrame

        :param target_col The column name of the target time series that needs to be modeled.
        All other columns will be considered as exogenous variables (if applicable to method)
        :type target_col str

        :param cv: Number of folds to use for cross validation.
        Number of observations in the Validation set for each fold = forecast period
        If None, a single fold is used
        :type cv Optional[int]

        :rtype object
        """


    @abstractmethod
    def refit(self, ts_df: pd.DataFrame) -> object:
        """
        Refits an already trained model using a new dataset
        Useful when fitting to the full data after testing with cross validation
        :param ts_df The time series data to be used for fitting the model
        :type ts_df pd.DataFrame
        :rtype object
        """

    @abstractmethod
    def predict(
        self,
        testdata: Optional[pd.DataFrame]=None,
        forecast_period: Optional[int] = None,
        simple: bool = True) -> NDFrame:
        """
        Return the predictions
        :param testdata The test dataframe containing the exogenous varaiables to be used for predicton.
        :type testdata Optional[pd.DataFrame]
        :param forecast_period The number of periods to make a prediction for.
        :type forecast_period Optional[int]
        :param simple If True, this method just returns the predictions.
        If False, it will return the standard error, lower and upper confidence interval (if available)
        :type simple bool
        :rtype NDFrame
        """

    def check_model_built(self):
        if self.model is None:
            raise AttributeError(
                "You are trying to perform an operation that requires the model to have been fit."+
                "However the model has not been fit yet. Please fit the model once before you try this operation."
            )

    def get_num_folds_from_cv(self, cv):
        if cv is None:
            NFOLDS = 1
        else:
            NFOLDS = cv

        return NFOLDS
