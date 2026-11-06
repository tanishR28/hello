def predict_proba(sentences):
    """LIME passes a LIST of strings. Return array (n_sentences, 2) of probabilities."""
    if isinstance(sentences, str):
        sentences = [sentences]
    enc = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=1).numpy()
