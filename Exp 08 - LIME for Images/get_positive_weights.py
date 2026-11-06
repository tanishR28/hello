def get_positive_weights(explanation, class_id):
    pairs = explanation.local_exp[class_id]                  # [(segment_id, weight), ...]
    positive = [(int(s), float(w)) for s, w in pairs if w > 0]
    return sorted(positive, key=lambda p: p[1], reverse=True)   # strongest first
