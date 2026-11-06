def lrp(x, W1, b1, W2, b2, eps=0.0):
    a1, logit = forward(x, W1, b1, W2, b2)
    R_output = logit                                   # start: relevance = the output score
 
    # output -> hidden : share the score according to each unit's contribution a1[k]*W2[k]
    z_hidden = a1 * W2[:, 0]
    R_hidden = z_hidden / (z_hidden.sum() + eps) * R_output
 
    # hidden -> input : each hidden unit shares ITS relevance among the words
    R_words = np.zeros_like(x)
    for k in range(W1.shape[1]):
        if a1[k] <= 0:                                 # dead ReLU unit: nothing to pass down
            continue
        z_words = x * W1[:, k]                         # contribution of every word to unit k
        R_words += z_words / (z_words.sum() + eps) * R_hidden[k]
    return R_words, R_hidden, logit
