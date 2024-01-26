import numpy as np
from sklearn.cluster import KMeans
import itertools
from sklearn.linear_model import Ridge
import logging
import sys
from scipy.stats import pearsonr
import matplotlib.colors as mcolors


logging.basicConfig(stream=sys.stdout, format="[%(asctime)s][%(levelname)s] %(message)s")


class ConceptLIME:

    '''
    self.current_expanation = {
        'idx': index of the explained image in self.dataset,
        'train_x': the training set of the surrogate model,
        'train_y': prediction of the surrogate model on its training set,
        'img': explained image in numpy array,
        'img_pred': predicted DISTRIBUTION of the explained model on the explained image,
        'img_top_cls': predicted class of the explained model on the explained image,
        'segment_mask': each pixel's segment id,
        'concept_mask': each pixel's concept id,
        'unique_cid': all different concept present in the explained image
    }
    '''

    def __init__(self, 
                 exp_model, 
                 dataset, 
                 seg_model,
                 perturb_mode="average",
                 perturb_color="white"
                 ):
        self.dataset = dataset
        self.model = exp_model
        self.seg_model = seg_model
        self.perturb_mode = perturb_mode


    def explain(self, 
                img_idx: int, 
                neighbor_size=100,
                perturbation_cnt=1000) -> np.ndarray:
        self.current_expanation = {}
        logging.info(f"explaining image {img_idx}")
        logging.info("selecting neighbors")
        neib_img_idxs = self._select_neighbor(img_idx, neighbor_size)
        logging.info("segmenting neighbors")
        neib_img_segs, masks = self._segment_images(neib_img_idxs)
        logging.info("making predictions on segments")
        seg_vectors = self._predict_segs(neib_img_segs)
        logging.info("clustering")
        seg_clusters = self._cluster(seg_vectors)
        
        logging.info("getting perturbation set and predicting on perturbation set")
        explained_img = np.asarray(self.seg_model.resize(self.dataset[img_idx]))
        train_x, perturb_pred = self._get_perturb_set(len(neib_img_segs[0]), seg_clusters, perturbation_cnt, masks[0], explained_img)
        img_pred = self.model.predict_prob(explained_img[np.newaxis, :])[0]
        img_top_cls = np.argmax(img_pred)
        train_y = perturb_pred[:, img_top_cls]
        logging.info("training surrogate model")
        surrog_model = Ridge(alpha=1, fit_intercept=True, random_state=0)
        surrog_model.fit(train_x, train_y) # type: ignore
        self.surrogate_model = surrog_model
        logging.info("finished")
        self.current_expanation['idx'] = img_idx
        self.current_expanation['train_x'] = train_x
        self.current_expanation['train_y'] = train_y
        self.current_expanation['img'] = explained_img
        self.current_expanation['img_pred'] = img_pred
        self.current_expanation['img_top_cls'] = img_top_cls
        self.current_expanation['segment_mask'] = masks[0]
        logging.getLogger().handlers[0].flush()
        return surrog_model.coef_
    

    def eval_fidelity(self,
                      color="average",
                      comb_thresh=100,):
        seg_ids = self.current_expanation['unique_cid']

        nn_pred_orig = self.current_expanation['img_pred'].max()
        surrog_pred_orig = self.surrogate_model.predict(self.current_expanation['train_x'])[-1]

        cpt_cnt = len(self.current_expanation['unique_cid'])

        correlations = {}
        for remove_cpt_cnt in range(1, len(seg_ids)):
            combinations = list(itertools.combinations(
                range(cpt_cnt), remove_cpt_cnt))

            if len(combinations) > comb_thresh:
                idxs = np.random.choice(list(range(len(combinations))), comb_thresh, False)
                combinations = np.array(combinations)[idxs]
            
            cpt_existence = np.ones((len(combinations), cpt_cnt))
            for i, combination in enumerate(combinations):
                cpt_existence[i, combination] = 0

            cpt_mask = self.current_expanation['concept_mask']
            _, cpt_idx_mask = np.unique(cpt_mask, return_inverse=True)
            cpt_idx_mask = cpt_idx_mask.reshape(cpt_mask.shape[0], cpt_mask.shape[1])
            image_pert = ConceptLIME._mask_imgs(cpt_existence, cpt_idx_mask, self.current_expanation['img'], color=color)

            nn_pred_pert = self.model.predict_prob(image_pert)[:, self.current_expanation['img_top_cls']]
            surrog_pred_pert = self.surrogate_model.predict(cpt_existence)

            correlations[remove_cpt_cnt] = pearsonr(
                np.repeat(nn_pred_orig, len(combinations)) - nn_pred_pert, 
                np.repeat(surrog_pred_orig, len(combinations)) - surrog_pred_pert)
            
        return correlations


    def _get_perturb_set(self, 
                         seg_cnt: int, 
                         seg_clusters: np.ndarray, 
                         size: int, 
                         seg_mask: np.ndarray, 
                         image: np.ndarray,
                         batch_size=64) -> 'tuple[np.ndarray, np.ndarray]':
        # explained image is at the tail of the samples
        cluster_ids = seg_clusters[range(seg_cnt)]
        self.current_expanation['unique_cid'] = np.unique(cluster_ids)
        combinations = np.array(list(itertools.product([0, 1], repeat=len(self.current_expanation['unique_cid']))))
        if len(combinations) > size:
            idxs = np.random.choice(list(range(len(combinations) - 1)), size - 1, False)
            sampled_items = combinations[idxs]
            # sampled_items = np.random.choice(combinations[:-1], size - 1, False)
            combinations = np.concatenate((sampled_items, combinations[-1].reshape(1, -1)))
        _, seg_idx_mask = np.unique(seg_mask, return_inverse=True)
        seg_idx_mask = seg_idx_mask.reshape(seg_mask.shape[0], seg_mask.shape[1])
        cpt_mask = seg_clusters[seg_idx_mask]
        _, cpt_idx_mask = np.unique(cpt_mask, return_inverse=True)
        cpt_idx_mask = cpt_idx_mask.reshape(cpt_mask.shape[0], cpt_mask.shape[1])
        self.current_expanation['concept_mask'] = cpt_mask
        perturb_pred = []
        for i in range(0, len(combinations), batch_size):
            temp_combinations = combinations[i:i+batch_size]
            if len(temp_combinations.shape) == 1:
                temp_combinations = temp_combinations.reshape(1, -1)
            temp_imgs = ConceptLIME._mask_imgs(temp_combinations, cpt_idx_mask, image)
            image_pred = self.model.predict_prob(temp_imgs, flag_hook=False)
            perturb_pred.extend(image_pred)
        return combinations, np.stack(perturb_pred)
    

    @staticmethod
    def _mask_imgs(zs: np.ndarray, 
                   segmentation: np.ndarray, 
                   image: np.ndarray,
                   color="average") -> np.ndarray:
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i,:,:,:] = image
            for j in range(zs.shape[1]):
                if zs[i,j] == 0:
                    if color == "average":
                        out[i][segmentation == j,:] = np.sum(image[segmentation == j], axis=(0)) / np.sum(segmentation == j)
                    else:
                        out[i][segmentation == j,:] = np.array(mcolors.to_rgb(color)) * 255
        return out
    
    
    def _cluster(self, feat_vecs: np.ndarray, k=None) -> np.ndarray:
        if not k:
            num_segs = len(feat_vecs)
            k = int(num_segs**0.5)
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=1).fit(feat_vecs)
        return kmeans.labels_


    def _predict_segs(self, img_segs: list) -> np.ndarray:
        self.model.outputs = []
        for img_seg in img_segs:
            for seg in img_seg:
                self.model.predict_logit(np.array(np.array(seg)[np.newaxis, :]), True)
        return np.array(self.model.outputs)


    def _segment_images(self, image_idxs: np.ndarray) -> 'tuple[list, list]':
        img_segs = []
        img_segs_masks = []
        for image_idx in image_idxs:
            segs, masks = self.seg_model.segment(self.dataset[image_idx])
            img_segs.append(segs)
            img_segs_masks.append(masks)
        return img_segs, img_segs_masks


    def _select_neighbor(self, 
                         img_idx: int, 
                         neighbor_size: int) -> np.ndarray:
        # explained image is at the head of the neighbors
        idxs = np.where(self.dataset.label_idx_PLACES == self.dataset.label_idx_PLACES[img_idx])[0]
        idxs_rm = np.delete(idxs, np.where(idxs == img_idx))
        idxs_select = np.random.choice(idxs_rm, neighbor_size - 1) if len(idxs) > neighbor_size - 1 else idxs_rm
        return np.insert(idxs_select, 0, img_idx)
        

    def eval_predict(self):
        return self.current_expanation['train_y'], self.surrogate_model.predict(self.current_expanation['train_x'])


    @staticmethod
    def generate_combinations(rangee, remove_cpt_cnt, comb_num):
            combinations = []
            while len(combinations) < comb_num:
                combination = np.random.choice(rangee, remove_cpt_cnt)
                if combination not in combinations:
                    combinations.append(combination)
            return combinations
    
