import numpy as np

def evaluate_model(model,it):
    correct=0
    total=0
    model.eval()
    for sentences_word, tags, lengths, mask in it:
        # set gradients zero
        model.zero_grad()
        # run model
        tag_scores = np.argmax(model(sentences_word, lengths).data.cpu().numpy(), axis=1)
        tag_gt = model.prepare_targets(tags, lengths).data.cpu().numpy()

        correct += np.sum(tag_scores == tag_gt)
        total += tag_gt.shape[0]

    return correct / total
