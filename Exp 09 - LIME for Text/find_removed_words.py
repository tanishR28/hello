def find_removed_words(original, changed):
    orig_words, changed_words = get_words(original), get_words(changed)
    removed, j = [], 0
    for w in orig_words:
        if j < len(changed_words) and w.lower() == changed_words[j].lower():
            j += 1                                          # word is still there
        else:
            removed.append(w)                               # word disappeared
    return removed
