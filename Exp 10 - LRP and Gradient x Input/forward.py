def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1                    # hidden pre-activation
    a1 = np.maximum(0, z1)              # ReLU
    logit = float(a1 @ W2[:, 0] + b2[0])    # output score (before sigmoid)
    return a1, logit
