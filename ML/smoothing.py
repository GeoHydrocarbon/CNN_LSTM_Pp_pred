import numpy as np
from scipy.signal import savgol_filter
from denoise import apply_median_filter_series


def choose_odd_window_length(num_points):
    """
    根据数据点数自适应选择奇数窗口长度：目标约为 N/50，限制在[5, 101]范围内且为奇数。
    """
    if num_points <= 5:
        return 5
    target = max(5, int(num_points / 50))
    if target % 2 == 0:
        target += 1
    target = min(target, 101)
    if target < 5:
        target = 5
    if target % 2 == 0:
        target += 1
    return target


def smooth_series(values, method="savgol", window_length=None, polyorder=2, ma_window=None, median_kernel=None):
    arr = np.asarray(values)
    n = len(arr)
    if n == 0:
        return arr
    if method == "savgol":
        wl = window_length if window_length is not None else choose_odd_window_length(n)
        wl = min(wl, n if n % 2 == 1 else n - 1)
        if wl < 3:
            wl = 3
        if wl % 2 == 0:
            wl += 1
        po = min(polyorder, wl - 1)
        try:
            return savgol_filter(arr, wl, po, mode="interp")
        except Exception:
            return arr
    elif method == "ma":
        w = ma_window if ma_window is not None else max(5, int(n / 50))
        if w % 2 == 0:
            w += 1
        w = max(3, min(w, n))
        if w <= 1:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")
    elif method == "median":
        return apply_median_filter_series(arr, kernel_size=median_kernel)
    else:
        return arr


def smooth_predictions_dataframe(predictions_df, model_names, method="savgol", window_length=None, polyorder=2, ma_window=None, median_kernel=None):
    df_sm = predictions_df.copy()
    for model_name in model_names:
        if model_name in df_sm.columns:
            df_sm[model_name] = smooth_series(
                df_sm[model_name].values,
                method=method,
                window_length=window_length,
                polyorder=polyorder,
                ma_window=ma_window,
                median_kernel=median_kernel,
            )
    return df_sm


