def ice_curves(model, X, feature, grid):
    """One row per sample, one column per grid value."""
    curves = []
    for value in grid:
        X_temp = X.copy()
        X_temp[feature] = value                  # set the feature to the grid value for ALL samples
        curves.append(model.predict(X_temp))     # one prediction per sample
    return np.array(curves).T                    # shape: (n_samples, n_grid)
