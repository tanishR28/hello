def deletion_test(im, explanation, class_id, base_prob, k_list=(1, 3, 5, 10), n_random=20):
    segments = explanation.segments
    weights = get_positive_weights(explanation, class_id)
    all_ids = np.unique(segments)
    rng = np.random.default_rng(42)
    results = []
    for k in k_list:
        if len(weights) < k:
            continue
        top_ids = [s for s, w in weights[:k]]                # LIME's k best regions
        deleted = im.copy()
        deleted[np.isin(segments, top_ids)] = 0              # blacken them
        lime_drop = base_prob - classifier_fn([deleted])[0, class_id]
 
        random_drops = []
        for _ in range(n_random):                            # same number of RANDOM regions
            ids = rng.choice(all_ids, size=k, replace=False)
            d = im.copy()
            d[np.isin(segments, ids)] = 0
            random_drops.append(base_prob - classifier_fn([d])[0, class_id])
        results.append((k, lime_drop, np.mean(random_drops)))
    return results
