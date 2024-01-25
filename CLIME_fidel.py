
import numpy as np
from Args import Args
from Utils.ModelResNet50Places365 import ResNet50_Places365
from Utils.DataSetADE20K import ADE20KDataset
from Utils.ModelXception65ADE import Xception65ADE
from Method.ConceptLIME import ConceptLIME
from Utils.utils import save_pickle


args = Args()
np.random.seed(args.SEED)


# Load ADE20K dataset
dataset = ADE20KDataset(args.get_path("ADE"))


# Load model to be explained
# device = torch.device("cpu")
explained_model = ResNet50_Places365(args.DEVICE)


# Load segmentation model
seg_model = Xception65ADE(args.get_path("XCEPTION_ADE"))


# explain
clime_explainer = ConceptLIME(explained_model, dataset, seg_model)


img_ids, _ = dataset.get_test_samples(50, 10, 10)
result = []
for img_id in img_ids:
    explanation = clime_explainer.explain(img_id, perturbation_cnt=args.SURROGATE_TRAINING_SIZE)
    correlations = clime_explainer.eval_fidelity(comb_thresh=args.FIDELITY_EVAL_SIZE)
    result.append(correlations)


save_pickle(f"{args.DIR_RESULT}/CLIME_fidel.pickle", result)
