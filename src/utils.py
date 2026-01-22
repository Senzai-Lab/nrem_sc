import numpy as np

def group_by_ids(values: np.ndarray, ids: np.ndarray, select_ids=None) -> dict:
    """
    Groups values by their corresponding ids.
    
    Parameters
    ----------
    values : np.ndarray
        1D array of values.
    ids : np.ndarray
        1D array of ids corresponding to each value in values.
    select_ids : array-like, optional
        Specific ids to group by. If None, all unique ids in ids are used.
    """
    assert values.ndim == 1, "values must be a 1D array"
    assert ids.ndim == 1, "ids must be a 1D array"
    assert values.shape[0] == ids.shape[0], "values and ids must have the same length"

    if select_ids is None:
        select_ids = np.unique(ids)
    return {uid: values[ids == uid] for uid in select_ids}