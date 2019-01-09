def get_predict_response(tokens, probs, preds, attn_out):
    """
        Args:
            tokens:
            probs:
            preds:
            attn_out:
        Returns:

    """
    if isinstance(tokens[0], str):
        tokens = [tokens]

    # TODO: Wrapping lists to prep for multiple contexts -> this will break so 100% needs to be fixed before release.

    resp_iterable = zip(tokens, probs, preds, attn_out)
    data = []

    for tokes, prob, label, attn_weights in resp_iterable:
        data.append({
            'tokens': tokes,
            # 'attentionWeights': attn_weights[:len(tokes)],
            'attentionWeights': attn_weights,
            'probs': prob,
            'label': label,
        })

    return {
        'numPredictions': len(preds),
        'data': data,
    }
