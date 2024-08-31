def calculate_rmse_imputed(original, imputed, missing_mask):
    original = np.array(original)
    imputed = np.array(imputed)
    missing_mask = np.array(missing_mask)
    squared_diff = np.square(original - imputed) * missing_mask
    sum_squared_diff = np.sum(squared_diff)
    num_imputed = np.sum(missing_mask)
    mse = sum_squared_diff / num_imputed
    rmse = np.sqrt(mse)
    var_squared_diff = np.var(squared_diff[missing_mask.astype(bool)])
    var_rmse = var_squared_diff / (4 * mse * num_imputed)
    ci_2sigma = 2 * np.sqrt(var_rmse)

    return [rmse, ci_2sigma]
