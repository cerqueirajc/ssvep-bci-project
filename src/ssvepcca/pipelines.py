from ssvepcca.utils import check_input_data, eval_accuracy, load_mat_data_array
from ssvepcca.definitions import NUM_BLOCKS, NUM_TARGETS, NUM_SUBJECTS

import os
import numpy as np
import toolz as fp


@fp.curry
def k_fold_predict(data, learner):
    """
    leave_one_out_predict
    ------------
    This function is a pipeline to be used with learners that needs training data. The proposed method is to fit
    the model with k-1 folds and predict the fold that was left out. This function is hardcoded to use NUM_BLOCKS
    as the number of folds (k=NUM_BLOCKS).
    """

    check_input_data(data)

    valid_masks = np.identity(NUM_BLOCKS, dtype=bool)
    train_masks = ~valid_masks

    predictions = np.empty([NUM_BLOCKS, NUM_TARGETS])
    predict_proba_list = []

    for block in range(NUM_BLOCKS):
        
        train_data = data[train_masks[block], :, :, :]
        valid_data = data[valid_masks[block], :, :, :].squeeze()
        
        learner.fit(train_data)
        
        for target in range(NUM_TARGETS):
            prediction, predict_proba = learner.predict(valid_data[target, :, :])

            predictions[block, target] = prediction
            predict_proba_list.append(predict_proba)

    return predictions, np.array(predict_proba_list), eval_accuracy(predictions)


@fp.curry
def test_fit_predict(data, learner):
    """
    test_fit_predict
    ----------
    This function is a pipeline to be used with learners that don't need training data (unsupervised) and, therefore,
    are fitted using the test data only. For example, classical CCA algorithm is applied to train data only.
    """

    check_input_data(data)

    predictions = np.empty([NUM_BLOCKS, NUM_TARGETS])
    predict_proba_list = []

    for block in range(NUM_BLOCKS):
        predict_proba_list.append([])
        for target in range(NUM_TARGETS):
            
            score_data = data[block, target, :, :]
            prediction, predict_proba = learner.predict(score_data)
            
            predictions[block, target] = prediction
            predict_proba_list[block].append(predict_proba)

    return predictions, np.array(predict_proba_list), eval_accuracy(predictions) # preds, pred_proba, acc


def eval_all_subjects_and_save_pipeline(learner_obj, fit_pipeline, dataset_root_path, output_folder):

    print(f"Run pipeline to evaluate the performance of an algorithm for all subjects and save assets")

    results = dict(
        predictions = [],
        accuracy = [],
        predict_proba = []
    )

    for subject_num in range(1, NUM_SUBJECTS + 1):

        print(f"Running evalulation for subject {subject_num}.")

        dataset = load_mat_data_array(f"{dataset_root_path}/S{subject_num}.mat")

        predictions, predict_proba, accuracy = fit_pipeline(dataset, learner_obj)

        results["predictions"].append(predictions)
        results["accuracy"].append(accuracy)
        results["predict_proba"].append(predict_proba)

    os.makedirs(output_folder, exist_ok=True)
    for name, result_array in results.items():
        np.save(output_folder + f"/{name}", np.array(result_array), allow_pickle=False)

    return
