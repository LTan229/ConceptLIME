from Args import Args
from Utils.ModelResNet50Places365 import ResNet50_Places365
from Utils.DataSetADE20K import ADE20KDataset
from Utils.ModelXception65ADE import Xception65ADE
from Method.ConceptLIME import ConceptLIME
from Utils.utils import save_pickle
import numpy as np

args = Args()
np.random.seed(args.SEED)

# Load ADE20K dataset
dataset = ADE20KDataset(args.get_path("ADE"))

# Load model to be explained
explained_model = ResNet50_Places365(args.DEVICE)

# Load segmentation model
seg_model = Xception65ADE(args.get_path("XCEPTION_ADE"))

# explain
clime_explainer = ConceptLIME(explained_model, dataset, seg_model)

img_ids, _ = dataset.get_test_samples(50, 10, 10)
result = {"target": [],
          "pred": []
          }
for img_id in img_ids:
    explanation = clime_explainer.explain(img_id, perturbation_cnt=args.SURROGATE_TRAINING_SIZE)
    result["target"].append(clime_explainer.current_expanation['train_y'].reshape(-1))
    
    result["pred"].append(clime_explainer.surrogate_model.predict(clime_explainer.current_expanation['train_x']).reshape(-1))

save_pickle(f"{args.DIR_RESULT}/CLIME_pred.pickle", result)

