def get_words(sentence):
    return re.findall(r"\b[\w'-]+\b", sentence)            # words without punctuation
