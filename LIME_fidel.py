"""
IMPORTANT:
    You need to make some modification to the marcotcr/lime library to enable the evaluation of fidelity.
    The modification allows the user to access the trained surrogated model.
    For details, please refer to the README.md in Method folder.
"""

import os
os.environ["TQDM_MININTERVAL"] = "5"


import itertools
from Args import Args
from Utils.ModelResNet50Places365 import ResNet50_Places365
from Utils.DataSetADE20K import ADE20KDataset
import numpy as np
from Method.lime import lime_image
from Utils.utils import mask_imgs, save_pickle
from scipy.stats import pearsonr
import math
import logging
import sys


logging.basicConfig(stream=sys.stdout, format="[%(asctime)s][%(levelname)s] %(message)s")


args = Args()
np.random.seed(args.SEED)


# Load ADE20K dataset
dataset = ADE20KDataset(args.get_path("ADE"))


# Load model to be explained
# device = torch.device("cpu")
model = ResNet50_Places365(args.DEVICE)


# util
def generate_combinations(comb_len, remove_cnt, comb_num):
    if math.comb(comb_len, remove_cnt) < 3 * comb_num:
        combinations = []
        for ids in itertools.combinations(range(comb_len), remove_cnt):
            combination = np.ones(comb_len)
            combination[list(ids)] = 0
            combinations.append(combination)
        combinations = np.array(combinations)
        if len(combinations) > comb_num:
            ids = np.random.choice(len(combinations), comb_num, replace=False)
            combinations = combinations[ids]
        return combinations
    else:
        combinations = []
        while len(combinations) < comb_num:
            ids = np.random.choice(comb_len, remove_cnt, replace=False)
            combination = np.ones(comb_len)
            combination[ids] = 0
            combination = combination.tolist()
            if combination not in combinations:
                combinations.append(combination)
        return np.array(combinations)


img_ids, _ = dataset.get_test_samples(50, 10, 10)
result = []
for img_id in img_ids:
    logging.info(f"running image {img_id}")
    explained_image = np.array(dataset[img_id])
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(explained_image, model.predict_prob, top_labels=1, hide_color=0, num_samples=args.SURROGATE_TRAINING_SIZE)
    logging.info("finish explanation")

    nn_pred = model.predict_prob(explained_image[np.newaxis, ...])
    nn_pred_cls = np.argmax(nn_pred)
    nn_pred_orig = nn_pred[0][nn_pred_cls]
    seg_cnt = np.unique(explanation.segments).shape[0]
    surrog_pred_orig = explainer.base.easy_model.predict(np.ones((1,seg_cnt)))[0]

    correlations = {}
    for remove_seg_cnt in range(10, seg_cnt, 10):
        logging.info(f"removing random combinations of {remove_seg_cnt} segments")
        combinations = generate_combinations(seg_cnt, remove_seg_cnt, args.FIDELITY_EVAL_SIZE)
        combinations = np.array(combinations)
        image_pert = mask_imgs(combinations, explanation.segments, explained_image, color="white")
        nn_pred_pert = model.predict_prob(image_pert)[:, nn_pred_cls]
        surrog_pred_pert = explainer.base.easy_model.predict(combinations)

        correlations[remove_seg_cnt] = pearsonr(
            np.repeat(nn_pred_orig, len(combinations)) - nn_pred_pert, 
            np.repeat(surrog_pred_orig, len(combinations)) - surrog_pred_pert
            )
    
    result.append(correlations)

save_pickle(f"{args.DIR_RESULT}/LIME_fidel.pickle", result)




