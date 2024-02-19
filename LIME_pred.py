import os
os.environ["TQDM_MININTERVAL"] = "5"


from Args import Args
from Utils.ModelResNet50Places365 import ResNet50_Places365
from Utils.DataSetADE20K import ADE20KDataset
import numpy as np
from lime import lime_image
from Utils.utils import save_pickle


args = Args()
np.random.seed(args.SEED)


# Load ADE20K dataset
dataset = ADE20KDataset(args.get_path("ADE"))


# Load model to be explained
args = Args()
# device = torch.device("cpu")
model = ResNet50_Places365(args.DEVICE)


img_ids, _ = dataset.get_test_samples(50, 10, 10)
result = {"target": [], # the prediction of the explained model on the explained image
          "pred": [], # the prediction of the surrogate model on the explained image
          "r2": [],# the r2 score on the training set of the surrogate model
          }
for img_id in img_ids:
    explained_image = np.array(dataset[img_id])
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        explained_image, model.predict_prob, top_labels=1, hide_color=0, num_samples=args.SURROGATE_TRAINING_SIZE
        )
    result["target"].append(model.predict_prob(explained_image[np.newaxis, ...]).max())
    result["pred"].append(explanation.local_pred[0])
    result["r2"].append(explanation.score)

save_pickle(f"{args.DIR_RESULT}/LIME_pred.pickle", result)

