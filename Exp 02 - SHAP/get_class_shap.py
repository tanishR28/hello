def get_class_shap(raw_shap, class_id=1):
    """Different SHAP versions return different shapes. Make it (samples x features)."""
    if isinstance(raw_shap, list):               # old SHAP: [class0_array, class1_array]
        return np.asarray(raw_shap[class_id])
    raw_shap = np.asarray(raw_shap)
    if raw_shap.ndim == 3:                       # new SHAP: (samples, features, classes)
        return raw_shap[:, :, class_id]
    return raw_shap                              # already 2-D
