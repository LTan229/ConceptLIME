import numpy as np
import tarfile
import tensorflow as tf
import os
from PIL import Image
import time

class Xception65ADE(object):

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        self.graph = tf.Graph()

        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        # tf.debugging.set_log_device_placement(True)
        resized_image = self.resize(image)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
            )
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


    def resize(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        return image.convert('RGB').resize(target_size, Image.LANCZOS)


    def segment(self, image): #, threshold=0.005):
        # each pixel is attributed an index that corresponds to certain class
        # the returned segments are sorted in the ascending order of that index due to the nature of np.unique
        resized_im, mask = self.run(image)
        segs = np.unique(mask)
        segments = []
        total = mask.shape[0] * mask.shape[1]
        for seg in segs:
            cur_mask = mask == seg
            sz = np.sum(cur_mask)
            # if sz < threshold * total:
            #     continue
            segment = resized_im * cur_mask[..., None]
            w, h = np.nonzero(cur_mask)
            segment = segment[np.min(w) : np.max(w) + 1, np.min(h) : np.max(h) + 1, :]
            segments.append(segment)
        return segments, mask
