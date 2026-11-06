def shap_importance(shap_vals, feature_names):
    mean_abs = np.abs(shap_vals).mean(axis=0)    # average |SHAP| over all samples
    table = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
    return table.sort_values("Mean |SHAP|", ascending=False)
