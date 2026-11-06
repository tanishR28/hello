def remove_one_word(sentence):
    words = get_words(sentence)
    original = predict_proba([sentence])[0][POS]            # positive probability now
    rows = []
    for i, w in enumerate(words):
        shorter = " ".join(words[:i] + words[i + 1:])       # sentence WITHOUT word i
        p = predict_proba([shorter])[0][POS]
        rows.append((w, p, original - p))                   # drop > 0  => word was pushing POSITIVE
    return sorted(rows, key=lambda r: r[2], reverse=True)
