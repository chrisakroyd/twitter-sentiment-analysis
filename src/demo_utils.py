def get_params(data, keys={'text', 'tokens'}):
    """ Creates dict of params from request restricted to a set of allowed keys."""
    return {key: data[key] for key in keys if key in data}


def get_predict_response(tokens, probs, preds, attn_out, orig_body):
    """
        Given tokens, probs, preds and attention weights generates a json formatted response body.
        Args:
            tokens: Tokenized form of the original text.
            probs: The softmax probabilities for each class.
            preds: The predicted class label.
            attn_out: Attention weights.
            orig_body: The original request body.
        Returns:
            Formatted prediction response message.
    """
    if isinstance(tokens[0], str):
        tokens = [tokens]

    # TODO: Wrapping lists to prep for multiple contexts -> this will break so 100% needs to be fixed before release.

    resp_iterable = zip(tokens, probs, preds, attn_out)
    data = []

    for tokes, prob, label, attn_weights in resp_iterable:
        data.append({
            'tokens': tokes,
            'attentionWeights': attn_weights,
            'probs': prob,
            'label': label,
        })

    return {
        'numPredictions': len(preds),
        'data': data,
        'parameters': get_params(orig_body)
    }


def get_error_response(error_message, orig_body, error_code=0):
    """
        Generates a formatted error response with the given error message and error code.
        Args:
            error_code: A unique id for this error.
            orig_body: The original parameters sent with the request which were invalid.
            error_message: Cause of this error.
        Returns:
            Error dict for the response.
    """
    return {
        'errorCode': error_code,
        'errorMessage': error_message,
        'parameters': get_params(orig_body)
    }
