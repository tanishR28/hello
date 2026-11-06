def find_worst_patch(image, sizes=(30, 50, 70, 90), step=30):
    candidates, batch = [], []
    for size in sizes:
        for y in range(30, 270, step):
            for x in range(30, 270, step):
                candidates.append((x, y, size))
                batch.append(preprocess_input(create_patched_image(image, x, y, size).copy()))
    preds = model.predict(np.array(batch), verbose=0)[:, target_class]   # confidence for every patch
    drops = original_conf - preds                           # how much confidence was lost
    best = int(np.argmax(drops))                            # patch that hurts the most
    return candidates[best], drops[best]
