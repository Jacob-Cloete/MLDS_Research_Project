def REMOVE(data, frac, mechanism='MCAR'):
    n, rows, cols = data.shape
    num_missing = int(rows * cols * frac)
    data_final = data.copy()
    for i in range(n):
        if mechanism == 'MCAR':
            missing_indices = np.random.choice(rows * cols, num_missing, replace=False)
        elif mechanism == 'MNAR':
            flat_image = data_final[i].flatten()
            sort_ind = np.argsort(flat_image)
            threshold_index = int(rows * cols * 0.5)
            above_thres_ind = sort_ind[threshold_index:]
            num_above_thres = len(above_thres_ind)
            num_miss_above_thres = min(num_missing, num_above_thres // 2)
            missing_indices = np.random.choice(above_thres_ind, num_miss_above_thres, replace=False)
        elif mechanism == 'MNAR1':
            square_rows, square_cols = rows // 4, cols // 4
            num_squares = 16
            num_squares_to_remove = min(16, max(1, round(frac * 16)))
            squares_to_remove = np.random.choice(num_squares, num_squares_to_remove, replace=False)
            missing_indices = []
            for square in squares_to_remove:
                start_row = (square // 4) * square_rows
                start_col = (square % 4) * square_cols
                for r in range(start_row, start_row + square_rows):
                    missing_indices.extend(range(r * cols + start_col, r * cols + start_col + square_cols))
            if len(missing_indices) > num_missing:
                missing_indices = np.random.choice(missing_indices, num_missing, replace=False)
        elif mechanism == 'MNAR2':
            num_strips = min(4, max(1, round(frac * 4)))
            strip_height = rows // 4
            strips_to_remove = np.random.choice(4, num_strips, replace=False)
            missing_indices = []
            for strip in strips_to_remove:
                start_row = strip * strip_height
                end_row = start_row + strip_height
                for row in range(start_row, end_row):
                    missing_indices.extend(range(row * cols, (row + 1) * cols))
            if len(missing_indices) > num_missing:
                missing_indices = np.random.choice(missing_indices, num_missing, replace=False)
        elif mechanism == 'MNAR3':
            square_rows, square_cols = 2, 2  
            num_squares = (rows // square_rows) * (cols // square_cols)  
            num_squares_to_remove = min(num_squares, max(1, round(frac * num_squares)))
            squares_to_remove = np.random.choice(num_squares, num_squares_to_remove, replace=False)
            missing_indices = []
            for square in squares_to_remove:
                start_row = (square // (cols // square_cols)) * square_rows
                start_col = (square % (cols // square_cols)) * square_cols
                for r in range(start_row, start_row + square_rows):
                    missing_indices.extend(range(r * cols + start_col, r * cols + start_col + square_cols))
            if len(missing_indices) > num_missing:
                missing_indices = np.random.choice(missing_indices, num_missing, replace=False)

        data_final[i, [idx // cols for idx in missing_indices], [idx % cols for idx in missing_indices]] = np.nan
    return data_final
