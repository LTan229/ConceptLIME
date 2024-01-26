import torchvision.models as models
import torch
from torchvision import transforms as trn
import numpy as np
from PIL import Image
from torch.autograd import Variable as V
import torch.nn as nn
import logging
import copy


class ResNet50_Places365():
    

    def __init__(self, device):
        self.device = device

        # load model
        arch = "resnet50"
        model_file = f"/home/ltan/LTan/Models/{arch}_places365.pth.tar" 
        self.model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {
            str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
        self.model.load_state_dict(state_dict)

        # convert model to evaluation mode with no grad
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # define centre crop
        self.centre_crop = trn.Compose(
            [
                trn.Resize((256, 256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                # trn.Normalize([0, 0, 0], [255, 255, 225]),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # attach hook
        self.outputs = [] # must be maunally maintained (cleared) when using
        def hook(module, input, output):
            x = nn.AdaptiveAvgPool2d(1)(output)
            self.outputs.append(x.cpu().numpy().squeeze())  # type: ignore
        self.model_hooked = copy.deepcopy(self.model)
        self.model_hooked.layer4[2].register_forward_hook(hook)
        self.model.to(device)
        self.model_hooked.to(device)


    def predict_prob(self, images: np.ndarray, flag_hook = False, batch_size=None) -> np.ndarray:
        if batch_size is not None:
            predictions = self.predict_logit_batch(images, flag_hook, batch_size)
        else:
            predictions = self.predict_logit(images, flag_hook)
        
        exp_predictions = np.exp(predictions)
        return exp_predictions / np.sum(exp_predictions, axis=1, keepdims=True)
    
    
    def predict_logit(self, images: np.ndarray, flag_hook = False) -> np.ndarray:
        input_images = self.prepossessing(images)
        if flag_hook:
            return self.model_hooked.forward(input_images).cpu().numpy()
        else:
            return self.model.forward(input_images).cpu().numpy()
        
    
    def predict_logit_batch(self, images: np.ndarray, flag_hook = False, batch_size=64) -> np.ndarray:
        input_images = self.prepossessing(images)

        logits = []
        for i in range(0, len(input_images), batch_size):
            batch = input_images[i:i+batch_size]
            if len(batch.shape) == 3:
                batch = batch.reshape(1, *batch.shape)

            if flag_hook:
                pred = self.model_hooked.forward(batch).cpu().numpy()
            else:
                pred = self.model.forward(batch).cpu().numpy()

            logits.extend(pred)
        return np.stack(logits)

    
    def prepossessing(self, images: np.ndarray) -> torch.Tensor:
        if images.max() <= 1:
            images = np.round(images * 255) # type: ignore
        images = np.uint8(images) # type: ignore
        results = []
        for image in images:
            img = Image.fromarray(image, 'RGB')
            input_img = V(self.centre_crop(img).unsqueeze(0)) # type: ignore
            results.append(input_img)
        return torch.cat(results).to(self.device)