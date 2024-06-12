import numpy as np

def decode_predictions(tokenizer, predictions):
    preds = predictions.predictions
    labels = predictions.label_ids

    if isinstance(preds, tuple):
            preds = preds[0]

    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    #preds = preds[:, 1:] #TODO: check if this is correct
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #labels = labels[:, 1:] #TODO: check if this is correct
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return {"labels": decoded_labels, "predictions": decoded_preds}
