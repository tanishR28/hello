def sensitivity(model, X, feature, delta):
    original = model.predict(X).mean()          # average prediction now
    X_new = X.copy()
    X_new[feature] = X_new[feature] + delta     # nudge the feature by delta
    new = model.predict(X_new).mean()           # average prediction after the nudge
    change = new - original
    return original, new, change, change / delta     # last one = sensitivity (change per unit)
