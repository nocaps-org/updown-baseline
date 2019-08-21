import numpy as np
import torch
from allennlp.data import Vocabulary
import h5py

BLACKLIST_CATEGORIES = [
            "Tree",
            "Building",
            "Plant",
            "Man",
            "Woman",
            "Person",
            "Boy",
            "Girl",
            "Human eye",
            "Skull",
            "Human head",
            "Human face",
            "Human mouth",
            "Human ear",
            "Human nose",
            "Human hair",
            "Human hand",
            "Human foot",
            "Human arm",
            "Human leg",
            "Human beard",
            "Human body",
            "Vehicle registration plate",
            "Wheel",
            "Seat belt",
            "Tire",
            "Bicycle wheel",
            "Auto part",
            "Door handle",
            "Clothing",
            "Footwear",
            "Fashion accessory",
            "Sports equipment",
            "Hiking equipment",
            "Mammal",
            "Personal care",
            "Bathroom accessory",
            "Plumbing fixture",
            "Land vehicle",
        ]

replacements = {
  "band-aid": "bandaid",
  "wood-burning stove": "wood burning stove",
  "kitchen & dining room table": "table",
  "salt and pepper shakers": "salt and pepper",
  "power plugs and sockets": "power plugs",
  "luggage and bags": 'luggage'
}

class cbs_matrix:

    def __init__(self, vocab_size):
        self.matrix = None
        self.vocab_size = vocab_size

    def init_matrix(self, state_size):
        self.matrix = np.zeros((1, state_size, state_size, self.vocab_size), dtype=np.uint8)

    def add_connect(self, from_state, to_state, w_group):
        assert self.matrix is not None
        for w_index in w_group:
            self.matrix[0, from_state, to_state, w_index] = 1
            self.matrix[0, from_state, from_state, w_index] = 0

    def init_row(self, state_index):
        assert self.matrix is not None
        self.matrix[0, state_index, state_index, :] = 1

    def get_matrix(self):
        return self.matrix

def suppress_parts(dets, classes):
    # just remove those 39 words
    keep = [i for i,cls in enumerate(classes) if cls not in BLACKLIST_CATEGORIES]
    return keep

class CBSConstraint(object):

    def __init__(self, features_h5path: str, oi_class_path: str, oi_word_form_path: str, vocabulary: Vocabulary, topk: int = 3):
        self.features_h5path = features_h5path
        self.topk = topk
        self._vocabulary = vocabulary
        self.M = cbs_matrix(self._vocabulary.get_vocab_size())

        self.boxes_h5 = h5py.File(self.features_h5path, "r")
        image_id_np = np.array(self.boxes_h5["image_id"])
        self._map = {
            image_id_np[index]: index for index in range(image_id_np.shape[0])
        }

        self.oi_class_list = [None]
        with open(oi_class_path) as out:
            for line in out:
                self.oi_class_list.append(line.strip().split(',')[1])
        self.oi_word_form = {}
        with open(oi_word_form_path) as out:
            for line in out:
                line = line.strip()
                items = line.split('\t')
                self.oi_word_form[items[0]] = items[1].split(',')

        self.obj_num = {}

    def select_state_func(self, beam_prediction, beam_score, imageID):
        max_step = beam_prediction.size(-1)
        selected_indices = torch.argmax(beam_score[:, 4:8, 0], dim=1) 
        selected_indices += torch.arange(selected_indices.size(0), device=selected_indices.device, dtype=selected_indices.dtype) * 4
        top_two_beam_prediction = beam_prediction[:, 4:8, 0, :].contiguous().view(-1, max_step)
        top_two_beam_prediction = top_two_beam_prediction.index_select(0, selected_indices)

        top_two_beam_prediction = top_two_beam_prediction.cpu().detach().numpy()
        beam_prediction = beam_prediction.cpu().detach().numpy()
        imageID = imageID.cpu().detach().numpy().tolist()

        pred = []
        for i, ID in enumerate(imageID):
            label_num = self.obj_num[ID]
            if label_num >= 3:
                # Three labels must be satisfied together
                pred.append(beam_prediction[i, 7, 0, :])
            elif label_num >= 2:
                # Two labels must be satisfied together
                pred.append(top_two_beam_prediction[i])
            elif label_num >= 1:
                # one label must be satisfied together
                pred.append(beam_prediction[i, 1, 0, :])
            else:
                # no constraint, get original beam
                pred.append(beam_prediction[i, 0, 0, :])

        return torch.from_numpy(np.array(pred, dtype=np.int64)).to(beam_score.device)

    def get_state_matrix(self, image_id: int):
        i = self._map[image_id.item()]

        box = self.boxes_h5["boxes"][i]
        box_cls = self.boxes_h5["classes"][i]
        box_score = self.boxes_h5["scores"][i]

        keep = box_score > 0
        box = box[keep]
        box_cls = box_cls[keep]
        box_score = box_score[keep]

        keep = suppress_parts(box, [self.oi_class_list[cls_] for cls_ in box_cls])
        box = box[keep]
        box_cls = box_cls[keep]
        box_score = box_score[keep]

        anns = list(zip(box_score,box_cls))
        anns = sorted(anns, key=lambda x:x[0], reverse=True) 

        candidates = []
        obj_names = []
        for s, cls_idx in anns[:self.topk]: # Keep up to three classes
            text = self.oi_class_list[cls_idx].lower()
            if text not in obj_names:
                obj_names.append(text)
                # Individual hacks to match existing vocab, etc
                if text in replacements:
                    text = replacements[text]
                
                group_w = []
                for w in text.split():
                    if w in self.oi_word_form:
                        group_w += self.oi_word_form[w]
                    else:
                        group_w.append(w)
                group_w = [self._vocabulary.get_token_index(w) for w in group_w]
                candidates.append([v for v in group_w if v > 0])

        self.obj_num[image_id.item()] = len(obj_names)

        self.M.init_matrix(8)
        for i in range(8):
            self.M.init_row(i)

        level_mapping = [{3:5, 2:6}, {1:6, 3:4}, {1:5, 2:4}]
        for i, group_w in enumerate(candidates):
            self.M.add_connect(0, i + 1, group_w)
            self.M.add_connect(i + 4, 7, group_w)

            mapping = level_mapping[i]
            for j in range(1, 4):
                if j in mapping:
                    self.M.add_connect(j, mapping[j], group_w)

        return self.M.get_matrix()
            
class FreeConstraint:

	def __init__(self, output_size):
		self.M = cbs_matrix(output_size)

	def select_state_func(self, beam_prediction, beam_score, imageID):
		return beam_prediction[:, 0, 0]

	def get_state_matrix(self, image_id):
		self.M.init_matrix(1)
		self.M.init_row(0)
		return self.M.get_matrix()





