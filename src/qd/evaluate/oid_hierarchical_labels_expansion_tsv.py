from tqdm import tqdm
import argparse
import json
import os.path as op
import copy
from qd.tsv_io import tsv_reader, tsv_writer
import torch
from maskrcnn_benchmark.layers import nms as _box_nms


def _load_labels(labels):
    if type(labels) is str:
        return json.loads(labels)
    elif type(labels) is list:
        return labels
    else:
        raise ValueError("Unknown box label format, should be either a json string or a list")

def apply_per_class_nms(labels, nms_thresh=0.5):
    box_labels = _load_labels(labels)
    classes = set([box['class'] for box in box_labels])
    new_labels = []
    for cls in classes:
        labels = [box for box in box_labels if box['class'] == cls]
        if len(labels) > 1:
            boxes = torch.tensor([box['rect'] for box in labels],
                    dtype=torch.float32)
            scores = torch.tensor([box['conf'] for box in labels],
                    dtype=torch.float32)
            keep = _box_nms(boxes, scores, nms_thresh)
            new_labels += [labels[i] for i in keep]
        else:
            new_labels += labels

    return new_labels


def _update_dict(initial_dict, update):
  """Updates dictionary with update content.

  Args:
   initial_dict: initial dictionary.
   update: updated dictionary.
  """

  for key, value_list in update.items():
    if key in initial_dict:
      initial_dict[key].extend(value_list)
    else:
      initial_dict[key] = value_list


def build_plain_hierarchy(hierarchy, skip_root=False):
  """Expands tree hierarchy representation to parent-child dictionary.

  Args:
   hierarchy: labels hierarchy as JSON file.
   skip_root: if true skips root from the processing (done for the case when all
     classes under hierarchy are collected under virtual node).

  Returns:
    keyed_parent - dictionary of parent - all its children nodes.
    keyed_child  - dictionary of children - all its parent nodes
    children - all children of the current node.
  """
  all_children = []
  all_keyed_parent = {}
  all_keyed_child = {}
  if 'Subcategory' in hierarchy:
    for node in hierarchy['Subcategory']:
      keyed_parent, keyed_child, children = build_plain_hierarchy(node)
      # Update is not done through dict.update() since some children have multi-
      # ple parents in the hierarchy.
      _update_dict(all_keyed_parent, keyed_parent)
      _update_dict(all_keyed_child, keyed_child)
      all_children.extend(children)

  if not skip_root:
    all_keyed_parent[hierarchy['LabelName']] = all_children
    all_children = [hierarchy['LabelName']] + all_children
    for child, _ in all_keyed_child.items():
      all_keyed_child[child].append(hierarchy['LabelName'])
    all_keyed_child[hierarchy['LabelName']] = []

  return all_keyed_parent, all_keyed_child, all_children


class OIDHierarchicalLabelsExpansion(object):
  def __init__(self, hierarchy):
    self._hierarchy_keyed_parent, self._hierarchy_keyed_child, _ = (
        build_plain_hierarchy(hierarchy, skip_root=True))

  def expand_boxes_from_tsv(self, tsv_row):
    rects = json.loads(tsv_row)
    rects_expand = []
    for rect in rects:
      parent_nodes = self._hierarchy_keyed_child[rect['class']]
      for parent_node in parent_nodes:
        rect1 = copy.copy(rect)
        rect1['class'] = parent_node
#        if rect1 not in rects and rect1 not in rects_expand:  # avoid duplicated expansion (diff from Google)
        rects_expand.append(rect1)
    result = rects + rects_expand
    return json.dumps(result)

  def expand_labels_from_tsv(self, tsv_row):
    labels = json.loads(tsv_row)
    labels_expand = []
    for lab in labels:
      if lab['conf'] == 1:
        # positive labels extend to parents
        assert lab['class'] in self._hierarchy_keyed_child
        parent_nodes = self._hierarchy_keyed_child[lab['class']]
        for parent_node in parent_nodes:
          lab1 = copy.copy(lab)
          lab1['class'] = parent_node
#        if lab1 not in labels and lab1 not in labels_expand:  # avoid duplicated expansion (diff from Google)
          labels_expand.append(lab1)
      elif lab['conf'] == 0:
        # negative labels extend to children
        assert lab['class'] in self._hierarchy_keyed_parent
        child_nodes = self._hierarchy_keyed_parent[lab['class']]
        for child_node in child_nodes:
          lab1 = copy.copy(lab)
          lab1['class'] = child_node
#          if lab1 not in labels and lab1 not in labels_expand:  # avoid duplicated expansion (diff from Google)
          labels_expand.append(lab1)
      else:
          raise Exception()
    result = labels + labels_expand
    return json.dumps(result)


def expand_labels(tsv_file, imagelevel_label, json_hierarchy_file,
        save_file, save_imagelevel_label_file,
        has_image_label=False, apply_nms=False):
  with open(json_hierarchy_file) as f:
    hierarchy = json.load(f)
  expansion_generator = OIDHierarchicalLabelsExpansion(hierarchy)
  if has_image_label:
      from qd.tsv_io import TSVFile
      imagelevel_tsv = TSVFile(imagelevel_label)
      imagelevelrows_new = []
  def gen_rows():
    for i, row in tqdm(enumerate(tsv_reader(tsv_file))):
      if has_image_label:
        expanded_boxes = expansion_generator.expand_boxes_from_tsv(row[-1])
        expanded_boxes = _load_labels(expanded_boxes)
        if apply_nms:
          # attach the conf if there is no
          if len(expanded_boxes) > 0:
              has_conf = 'conf' in expanded_boxes[0]
              if has_conf:
                  assert all('conf' in x for x in expanded_boxes)
              else:
                  assert all('conf' not in x for x in expanded_boxes)
              if not has_conf:
                  for x in expanded_boxes:
                      x['conf'] = 1.
          expanded_boxes = apply_per_class_nms(expanded_boxes)
          expanded_boxes = _load_labels(expanded_boxes)
          if len(expanded_boxes) > 0:
              if not has_conf:
                  for x in expanded_boxes:
                      del x['conf']
        expanded_boxes = json.dumps(expanded_boxes)
        imagelevel_row = imagelevel_tsv[i]
        assert row[0] == imagelevel_row[0]
        expanded_labels = expansion_generator.expand_labels_from_tsv(imagelevel_row[1])
        imagelevelrows_new.append((row[0], expanded_labels))
        row_new = [row[0], expanded_boxes]
      else:
        expanded_boxes = expansion_generator.expand_boxes_from_tsv(row[-1])
        expanded_boxes = _load_labels(expanded_boxes)
        if apply_nms:
          expanded_boxes = apply_per_class_nms(expanded_boxes)
        expanded_boxes = json.dumps(expanded_boxes)
        row_new = [row[0], expanded_boxes]
      yield row_new

  tsv_writer(gen_rows(), save_file)

  if has_image_label:
    tsv_writer(imagelevelrows_new, save_imagelevel_label_file)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
      description='Hierarchically expand annotations (excluding root node).')
    parser.add_argument('--json_hierarchy_file', required=True,
      help='Path to the file containing label hierarchy in JSON format.')
    parser.add_argument('--tsv_file', required=True,
      help='Path to Open Images annotations file (either box-level or image-level labels)')
    parser.add_argument('--save_file', type=str, required=False, default=None,
                        help="""Output file name""")
    parser.add_argument('--has_image_label', required=False, type=bool, default=False,
                        help="expand image label or not")
    parser.add_argument('--apply_nms', required=False, default=False, action='store_true',
                        help='apply nms after label expansion to avoid duplicated expansion')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
  """ This script is adapted from Google's implementation of label expansion on 
  csv files. We extend it to work with tsv format. 
  
  Note: in Google's implementation, there is duplicated label expansion. That is 
        a label can be expanded to the same parent twice. (Shrimp->Shellfish->Seafood,
        Shrimp->Shellfish->..->Animal). By default, we follow Google's implementation 
        in order to produce the same result. 
        
        A second type of duplicated expansion happens when there is a detection of both 
        child and parent class. The expansion do not distinguish this case. So a new 
        feature of --apply_nms is added to perform per class nms after label expansion. 
  """

  args = parse_args()
  expand_labels(args.tsv_file, args.json_hierarchy_file, args.save_file, args.has_image_label, args.apply_nms)
