import torch.nn as nn


class SoftMaxTreePrediction(nn.Module):
    def __init__(self, tree, pred_thresh):
        super(SoftMaxTreePrediction, self).__init__()
        from mtorch.softmaxtree import SoftmaxTree
        from mtorch.softmaxtree_prediction import SoftmaxTreePrediction
        self.class_prob = SoftmaxTree(tree, axis=1)
        self.predictor = SoftmaxTreePrediction(
                tree,
                threshold=pred_thresh,
                append_max=False,
                output_tree_path=True,
                )

    def forward(self, x):
        x = self.class_prob(x)
        obj = None
        top_preds = self.predictor(x, obj)
        return top_preds

