def manual_pfi(model, X, y, n_repeats=1):
    baseline = accuracy_score(y, model.predict(X))      # 1. score BEFORE shuffling
    importance = []
    for feature in X.columns:                            # 2. one feature at a time
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()                            # 3. copy, keep original safe
            X_perm[feature] = np.random.permutation(X_perm[feature].values)  # 4. shuffle ONE column
            score = accuracy_score(y, model.predict(X_perm))   # 5. score AFTER shuffling
            drops.append(baseline - score)               # 6. importance = how much accuracy fell
        importance.append(np.mean(drops))
    return importance
