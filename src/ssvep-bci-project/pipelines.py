import numpy as np
from utils import check_input_data
from definitions import NUM_BLOCKS, NUM_TARGETS


def leave_one_out_predict(learner, data, target, electrode_index=None):
    """
    leave_one_out_predict
    ------------
    This function is a pipeline to be used with learners that needs training data. The proposed method is to fit
    the model with k-1 examples and predict the example left out.
    """

    check_input_data(data)
    
    if electrode_index:
        data = data[:, :, :, electrode_index]

    score_masks = np.identity(NUM_BLOCKS, dtype=bool)
    train_masks = ~score_masks
    
    output_predictions = np.empty([NUM_BLOCKS])
    
    for block in range(NUM_BLOCKS):
        train_data = data[:, :, train_masks[block]]
        score_data = data[:, :, score_masks[block]]
        
        learner.fit(train_data, target)
        output_predictions[block] = learner.predict(score_data)
    
    return output_predictions


def test_fit_predict(learner, data):
    """
    test_fit_predict
    ----------
    This function is a pipeline to be used with learners that don't need training data (unsupervised) and, therefore,
    are fitted using the test data only. For example, classical CCA algorithm is applied to train data only.
    """

    check_input_data(data)

    output_predictions = np.empty([NUM_BLOCKS, NUM_TARGETS])

    for block in range(NUM_BLOCKS):
        for target in range(NUM_TARGETS):
            score_data = data[block, target, :, :]
            output_predictions[block, target] = learner.predict(score_data)

    return output_predictions
