def IMPUTE(data_missing, imputation='mean', n_neighbors=5):
    n, rows, cols = data_missing.shape
    data_imputed = np.copy(data_missing)

    if imputation == 'mean':
        mean_values = np.nanmean(data_missing, axis=0)

        for i in range(n):
            mask = np.isnan(data_missing[i])
            data_imputed[i][mask] = mean_values[mask]
    elif imputation == 'random':
        for i in range(n):
            mask = np.isnan(data_missing[i])
            non_missing_values = data_missing[i][~mask]
            if len(non_missing_values) > 0:
                data_imputed[i][mask] = np.random.choice(non_missing_values, size=np.sum(mask))
            else:
                data_imputed[i][mask] = 0
    elif imputation == 'knn':
        for i in range(n):
            img = data_missing[i]
            missing_mask = np.isnan(img)
            img_zeros = np.where(missing_mask, 0, img)

            y, x = np.indices(img.shape)
            all_coords = np.column_stack((x.ravel(), y.ravel()))
            all_values = img_zeros.ravel()

            tree = cKDTree(all_coords)

            for j in range(len(all_coords)):
                if missing_mask.ravel()[j]:
                    distances, indices = tree.query(all_coords[j], k=n_neighbors+1)  # +1 to exclude self

                    distances = distances[1:]
                    indices = indices[1:]

                    neighbor_values = all_values[indices]
                    weights = 1 / (distances + 1e-8)
                    imputed_value = np.sum(weights * neighbor_values) / np.sum(weights)

                    data_imputed[i].ravel()[j] = imputed_value
                    all_values[j] = imputed_value

    return data_imputed
