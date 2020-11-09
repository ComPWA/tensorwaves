# cspell:ignore fillna

"""The data module takes care of data generation."""

__all__ = [
    "generate",
    "tf_phasespace",
    "PwaAccessor",
]

from typing import List, Optional

import numpy as np
import pandas as pd

from . import generate, tf_phasespace
from ._data_frame import ENERGY_LABEL, MOMENTUM_LABELS, WEIGHT_LABEL


@pd.api.extensions.register_dataframe_accessor("pwa")
class PwaAccessor:
    """`~pandas.DataFrame` accessor for PWA properties.

    Additional namespace to interpret a `~pandas.DataFrame` as 'PWA style'
    dataframe, see `here
    <https://pandas.pydata.org/pandas-docs/stable/development/extending.html#registering-custom-accessors>`_.
    """

    def __init__(self, pandas_object: pd.DataFrame) -> None:
        self._validate(pandas_object)
        self._obj = pandas_object

    @staticmethod
    def _validate(obj: pd.DataFrame) -> None:
        columns = obj.columns
        if isinstance(columns, pd.MultiIndex):
            # if multicolumn, test if 2 levels
            columns = columns.levels
            if len(obj.columns.levels) != 2:
                raise IOError(
                    "Not a PWA data data frame!\n"
                    "pandas.DataFrame must have multicolumns of 2 levels:\n"
                    " - 1st level are the particles names\n"
                    " - 2nd level define the 4-momentum:"
                    f"{MOMENTUM_LABELS}"
                )
            # then select 2nd columns only
            columns = columns[1]
        # Check if (sub)column names are same as momentum labels
        if not set(MOMENTUM_LABELS) <= set(columns):
            raise IOError(f"Columns must be {MOMENTUM_LABELS}")

    @property
    def weights(self) -> Optional[pd.DataFrame]:
        """Get list of weights, if available."""
        if WEIGHT_LABEL not in self.column_names:
            return None
        return self._obj[WEIGHT_LABEL]

    @property
    def intensities(self) -> Optional[pd.DataFrame]:
        """Alias for :func:`weights` in the case of a fit intensity sample."""
        return self.weights

    @property
    def column_names(self) -> list:
        """Get a list of the top layer column names."""
        columns = self._obj.columns
        if isinstance(columns, pd.MultiIndex):
            columns = self._obj.columns.droplevel(1).unique()
        return columns.to_list()

    @property
    def particles(self) -> Optional[List[str]]:
        """Get list of non-particle columns contained in the data frame."""
        columns = self._obj.columns
        if not isinstance(columns, pd.MultiIndex):
            return None
        return [
            col
            for col in self.column_names
            if isinstance(self._obj[col], pd.DataFrame)
            and self._obj[col].columns.unique().to_list() == MOMENTUM_LABELS
        ]

    @property
    def energy(self) -> pd.DataFrame:
        """Get a dataframe containing only the energies."""
        if isinstance(self._obj.columns, pd.MultiIndex):
            return self._obj.xs(ENERGY_LABEL, level=1, axis=1)
        return self._obj[ENERGY_LABEL]

    @property
    def p4sum(self) -> pd.DataFrame:
        """Get the total 4-momenta of the particles."""
        return self._obj.sum(
            axis=1,
            level=len(self._obj.columns.names) - 1,
        )

    @property
    def p_xyz(self) -> pd.DataFrame:
        """Get a dataframe containing only the 3-momenta."""
        return self._obj.filter(regex=("p_[xyz]"))

    @property
    def rho2(self) -> pd.DataFrame:
        """**Compute** quadratic sum of the 3-momenta."""
        if isinstance(self._obj.columns, pd.MultiIndex):
            return (self.p_xyz ** 2).sum(axis=1, level=0)
        return (self.p_xyz ** 2).sum(axis=1)

    @property
    def rho(self) -> pd.DataFrame:
        """**Compute** absolute value of the 3-momenta."""
        return np.sqrt(self.rho2)

    @property
    def mass2(self) -> pd.DataFrame:
        """**Compute** the square of the invariant masses."""
        mass2 = self.energy ** 2 - self.rho2
        return abs(mass2.fillna(0))

    @property
    def mass(self) -> pd.DataFrame:
        """**Compute** the invariant masses."""
        return np.sqrt(self.mass2)
