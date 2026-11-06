def partial_dependence_manual(model, X, feature, n_points=50):
    grid = np.linspace(X[feature].min(), X[feature].max(), n_points)   # 1. values to try
    pd_values = []
    for value in grid:
        X_temp = X.copy()                       # 2. fresh copy of the data
        X_temp[feature] = value                 # 3. force EVERY row to this value
        preds = model.predict(X_temp)           # 4. predict for all rows
        pd_values.append(preds.mean())          # 5. average prediction = PDP point
    return grid, np.array(pd_values)
