import os, copy
import numpy as np
from salt.onnx_model import OnnxModels
from salt.dataset_explorer import DatasetExplorer
from salt.display_utils import DisplayUtils


class CurrentCapturedInputs:
    def __init__(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def reset_inputs(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def set_mask(self, mask):
        self.curr_mask = mask

    def add_input_click(self, input_point, input_label):
        if len(self.input_point) == 0:
            self.input_point = np.array([input_point])
        else:
            self.input_point = np.vstack([self.input_point, np.array([input_point])])
        self.input_label = np.append(self.input_label, input_label)

    def set_low_res_logits(self, low_res_logits):
        self.low_res_logits = low_res_logits


class Editor:
    def __init__(self, onnx_models_path, dataset_path, categories=None, coco_json_path=None):
        self.dataset_path = dataset_path
        self.coco_json_path = coco_json_path
        if categories is None and not os.path.exists(coco_json_path):
            raise ValueError("categories must be provided if coco_json_path is None")
        if self.coco_json_path is None:
            self.coco_json_path = os.path.join(self.dataset_path, "annotations.json")
        self.dataset_explorer = DatasetExplorer(
            self.dataset_path, categories=categories, coco_json_path=self.coco_json_path
        )
        self.curr_inputs = CurrentCapturedInputs()
        self.categories, self.category_colors = self.dataset_explorer.get_categories(get_colors=True)
        self.color_by_tracker = False
        self.image_id = min(self.dataset_explorer.getImgIds(), default=0)
        self.category_id = 0
        self.show_other_anns = True
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self._set_prev_image_with_annotations()
        self.onnx_helper = OnnxModels(
            onnx_models_path,
            image_width=self.image.shape[1],
            image_height=self.image.shape[0],
        )
        self.du = DisplayUtils()
        self.reset()

    @property
    def next_tracker_id(self):
        return self.dataset_explorer.next_tracker_id

    def list_annotations(self):
        anns, colors = self.dataset_explorer.get_annotations(self.image_id, return_colors=True)
        return anns, colors

    def toggle_color_by_tracker(self):
        self.color_by_tracker = not self.color_by_tracker

    def delete_annotations(self, annotation_id):
        self.dataset_explorer.delete_annotations(annotation_id)

    def __draw_known_annotations(self, selected_annotations=[]):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True, color_by_tracker=self.color_by_tracker
        )

        for i, (ann, color) in enumerate(zip(anns, colors)):
            for selected_ann in selected_annotations:
                if ann["id"] == selected_ann:
                    colors[i] = (0, 0, 0)
        # Use this to list the annotations
        self.display = self.du.draw_annotations(self.display, anns, colors)

    def __draw(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        if self.curr_inputs.curr_mask is not None:
            self.display = self.du.draw_points(self.display, self.curr_inputs.input_point, self.curr_inputs.input_label)
            self.display = self.du.overlay_mask_on_image(self.display, self.curr_inputs.curr_mask)
        if self.show_other_anns:
            self.__draw_known_annotations(selected_annotations)

    def draw_selected_annotations(self, selected_annotations=[]):
        self.__draw(selected_annotations)

    def _set_prev_image_with_annotations(self):
        if not hasattr(self, "prev_display"):
            self._prev_image_bgr = np.zeros_like(self.image_bgr)
        if not hasattr(self, "prev_anns"):
            self.prev_anns = []
        if not hasattr(self, "prev_colors"):
            self.prev_colors = []

        if self.image_id == min(self.dataset_explorer.getImgIds(), default=0):
            self._prev_image_bgr = np.zeros_like(self.image_bgr)
            return

        image_id = self.image_id - 1
        _, self._prev_image_bgr, _ = self.dataset_explorer.get_image_data(image_id)

        self.prev_anns, self.prev_colors = self.dataset_explorer.get_annotations(
            image_id, return_colors=True, color_by_tracker=self.color_by_tracker
        )

    def draw_prev_image_with_annotations(self):
        """
        Returns the prev image with annotations drawn on it.
        If the current image is the last one, it returns a blank image.
        """
        display = self._prev_image_bgr.copy()
        if self.show_other_anns:
            display = self.du.draw_annotations(display, self.prev_anns, self.prev_colors)
        return display

    def add_click(self, new_pt, new_label, selected_annotations=[]):
        self.curr_inputs.add_input_click(new_pt, new_label)
        masks, low_res_logits = self.onnx_helper.call(
            self.image,
            self.image_embedding,
            self.curr_inputs.input_point,
            self.curr_inputs.input_label,
            low_res_logits=self.curr_inputs.low_res_logits,
        )
        self.curr_inputs.set_mask(masks[0, 0, :, :])
        self.curr_inputs.set_low_res_logits(low_res_logits)
        self.__draw(selected_annotations)

    def remove_click(self, new_pt):
        print("ran remove click")

    def reset(self, selected_annotations=[]):
        self.curr_inputs.reset_inputs()
        self.__draw(selected_annotations)

    def toggle(self, selected_annotations=[]):
        self.show_other_anns = not self.show_other_anns
        self.__draw(selected_annotations)

    def step_up_transparency(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        self.du.increase_transparency()
        self.__draw(selected_annotations)

    def step_down_transparency(self, selected_annotations=[]):
        self.display = self.image_bgr.copy()
        self.du.decrease_transparency()
        self.__draw(selected_annotations)

    def save_ann(self):
        self.dataset_explorer.add_annotation(self.image_id, self.category_id, self.curr_inputs.curr_mask)

    def save(self):
        self.dataset_explorer.save_annotation()

    def change_annotation_category(self, selected_annotations=[]):
        self.dataset_explorer.update_annotation_category(self.category_id, selected_annotations)
        self.__draw(selected_annotations)

    def change_annotation_tracker_id(self, selected_annotations=[], new_id=None):
        self.dataset_explorer.update_annotation_tracker_id(selected_annotations, new_id)
        self.__draw(selected_annotations)

    def go_to_image(self, image_id):
        if image_id < 0 or image_id >= self.dataset_explorer.get_num_images():
            return
        self.image_id = image_id
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0])
        self.reset()

    def next_image(self):
        if self.image_id == self.dataset_explorer.get_num_images() - 1:
            return
        self.image_id += 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0])
        self._set_prev_image_with_annotations()
        self.reset()

    def prev_image(self):
        if self.image_id == 0:
            return
        self.image_id -= 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.onnx_helper.set_image_resolution(self.image.shape[1], self.image.shape[0])
        self._set_prev_image_with_annotations()
        self.reset()

    def next_category(self):
        if self.category_id == len(self.categories) - 1:
            self.category_id = 0
            return
        self.category_id += 1

    def prev_category(self):
        if self.category_id == 0:
            self.category_id = len(self.categories) - 1
            return
        self.category_id -= 1

    def get_categories(self, get_colors=False):
        if get_colors:
            return self.categories, self.category_colors
        return self.categories

    def select_category(self, category_name):
        category_id = self.categories.index(category_name)
        self.category_id = category_id

    def sort_images(self):
        self.dataset_explorer.sort_images()
        self.image_id = min(self.dataset_explorer.getImgIds(), default=0)
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.reset()
