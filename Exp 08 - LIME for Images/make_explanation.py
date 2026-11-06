def make_explanation(im, segmentation_fn, num_samples=800, seed=42):
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    return explainer.explain_instance(
        im, classifier_fn, top_labels=3, hide_color=0,
        num_samples=num_samples, segmentation_fn=segmentation_fn, random_seed=seed)
