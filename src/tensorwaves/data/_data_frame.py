"""Functions for creating a correctly formatted `pandas.DataFrame`.

This module is a sub-module so that it is available for modules like
`.data.generate` without creating cycling dependencies. It offers functionality
to create `~.pandas.DataFrame` s that comply the :code:`_validate` method of
the `.PwaAccessor`.
"""


from typing import Iterable, Sequence

import pandas as pd

ENERGY_LABEL = "E"
MOMENTUM_LABELS = ["p_x", "p_y", "p_z", ENERGY_LABEL]
WEIGHT_LABEL = "weight"


def create_frame(
    data: Sequence,
    column_names: Iterable,
) -> pd.DataFrame:
    """Create an `PWA DataFrame <.PwaAccessor>`.

    The columns of the `~pandas.DataFrame` are specially formatted so that they
    agree with the :code:`_validate` method of the `.PwaAccessor`.
    """
    multi_column = create_multicolumn(column_names)
    return pd.DataFrame(
        data=data,
        columns=multi_column,
    )


def create_multicolumn(column_names: Iterable) -> pd.Index:
    """Create a multicolumn.

    The multicolumn complies with the complies with the standards set by the
    `~.PwaAccessor`.
    """
    cols = [
        (top_column, mom)
        for top_column in column_names
        for mom in MOMENTUM_LABELS
    ]
    return pd.MultiIndex.from_tuples(
        tuples=cols, names=["StateID", "Momentum"]
    )
