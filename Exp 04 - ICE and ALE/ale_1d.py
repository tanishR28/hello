def ale_1d(model, X, feature, n_bins=10):
    bins = np.linspace(X[feature].min(), X[feature].max(), n_bins + 1)   # 1. cut range into bins
    effects, centers = [], []
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == n_bins - 1:                                   # last bin includes the right edge
            mask = (X[feature] >= lower) & (X[feature] <= upper)
        else:
            mask = (X[feature] >= lower) & (X[feature] < upper)
        X_bin = X[mask]                                       # 2. ONLY the real rows inside this bin
        if len(X_bin) == 0:
            continue
        X_low, X_up = X_bin.copy(), X_bin.copy()
        X_low[feature] = lower                                # 3. same rows, feature at lower edge
        X_up[feature] = upper                                 #    same rows, feature at upper edge
        diff = model.predict(X_up) - model.predict(X_low)     # 4. local change in prediction
        effects.append(diff.mean())                           # 5. average local effect of this bin
        centers.append((lower + upper) / 2)
    ale = np.cumsum(effects)                                  # 6. accumulate the local effects
    ale = ale - ale.mean()                                    # 7. centre around zero
    return np.array(centers), ale
