import torch
from torch import nn
from qd.mask.structures.boxlist_ops import boxlist_nms_no_convert_back
from qd.mask.structures.boxlist_ops import boxlist_softnms

class BoxListSort(nn.Module):
    def __init__(self, max_proposals, score_field):
        super(BoxListSort, self).__init__()
        self.max_proposals = max_proposals
        self.score_field = score_field

    def forward(self, x):
        if len(x.bbox) <= self.max_proposals:
            return x
        _, idx = torch.topk(x.get_field(self.score_field),
                self.max_proposals)
        return x[idx]

class BoxListNMS(nn.Module):
    def __init__(self, thresh, max_proposals, score_field):
        super(BoxListNMS, self).__init__()
        self.thresh = thresh
        self.max_proposals = max_proposals
        self.score_field = score_field
        self.input_mode = 'xyxy'

    def forward(self, x):
        return  boxlist_nms_no_convert_back(x,
                self.thresh,
                max_proposals=self.max_proposals,
                score_field=self.score_field)

class BoxListSoftNMS(nn.Module):
    def __init__(self, thresh, max_proposals, score_field,
            score_thresh):
        super(BoxListSoftNMS, self).__init__()
        self.thresh = thresh
        self.score_field = score_field
        self.input_mode = 'xyxy'
        if max_proposals == -1:
            max_proposals = 10000000000;
        self.max_proposals = max_proposals
        self.score_thresh = score_thresh

    def forward(self, x):
        return boxlist_softnms(x, self.thresh,
                threshold=self.score_thresh,
                max_box=self.max_proposals,
                score_field=self.score_field)


class BoxListHNMS(nn.Module):
    def __init__(self, nms_policy, max_proposals, score_field):
        super().__init__()
        from hnms import MultiHNMS
        self.hnms = MultiHNMS(num=nms_policy.NUM, alpha=nms_policy.ALPHA)
        self.score_field = score_field
        self.max_proposals = max_proposals
        self.input_mode = 'cxywh'
    def forward(self, boxlist):
        boxlist = boxlist.convert(self.input_mode)
        rects = boxlist.bbox
        scores = boxlist.get_field(self.score_field)
        keep = self.hnms(rects, scores)
        if self.max_proposals > 0 and len(keep) > self.max_proposals:
            _, idx = scores[keep].topk(self.max_proposals)
            keep = keep[idx]
        boxlist = boxlist[keep]
        return boxlist

class BoxListComposeHNMS(nn.Module):
    def __init__(self, nms_policy, max_proposals, score_field):
        super().__init__()
        from qd.hnms import MultiHashNMSAnyKPt
        if nms_policy.NUM == 0:
            hnms1 = None
        else:
            hnms1 = MultiHashNMSAnyKPt(
                    num=nms_policy.NUM,
                    w0=nms_policy.WH0,
                    h0=nms_policy.WH0,
                    alpha=nms_policy.ALPHA,
                    gamma=nms_policy.GAMMA,
                    rerank=False)
        if nms_policy.NUM2 == 0:
            hnms2 = None
        else:
            hnms2 = MultiHashNMSAnyKPt(
                    num=nms_policy.NUM2,
                    w0=nms_policy.WH0,
                    h0=nms_policy.WH0,
                    alpha=nms_policy.ALPHA2,
                    gamma=nms_policy.GAMMA2,
                    rerank=True,
                    rerank_iou=nms_policy.THRESH2)
        if nms_policy.COMPOSE_FINAL_RERANK:
            if nms_policy.COMPOSE_FINAL_RERANK_TYPE == 'softnms':
                rerank = BoxListSoftNMS(
                    nms_policy.THRESH,
                    score_thresh=0.,
                    max_proposals=max_proposals,
                    score_field=score_field)
            elif nms_policy.COMPOSE_FINAL_RERANK_TYPE == 'nms':
                rerank = BoxListNMS(nms_policy.THRESH,
                        max_proposals=max_proposals,
                        score_field=score_field)
            elif nms_policy.COMPOSE_FINAL_RERANK_TYPE == 'sort':
                rerank = BoxListSort(max_proposals, score_field)
            else:
                raise NotImplementedError(nms_policy.COMPOSE_FINAL_RERANK_TYPE)
        else:
            rerank = None
        self.hnms1 = hnms1
        self.hnms2 = hnms2
        self.rerank = rerank
        self.nms_policy = nms_policy

        self.score_field = score_field
        self.max_proposals = max_proposals
        self.input_mode = 'cxywh'

    def forward(self, boxlist):
        if self.hnms1 is not None or self.hnms2 is not None:
            #origin_mode = boxlist.mode
            boxlist = boxlist.convert('cxywh')
            rects = boxlist.bbox
            scores = boxlist.get_field(self.score_field)
        if self.hnms1 is not None and self.hnms2 is not None:
            keep = self.hnms1(rects, scores)
            keep2 = self.hnms2(rects[keep], scores[keep])
            keep = keep[keep2]
        elif self.hnms1 is not None:
            keep = self.hnms1(rects, scores)
        elif self.hnms2 is not None:
            keep = self.hnms2(rects, scores)
        else:
            keep = None

        if self.rerank is not None:
            if keep is not None:
                boxlist = boxlist[keep]
            boxlist = self.rerank(boxlist)
        else:
            if self.max_proposals > 0 and len(keep) > self.max_proposals:
                _, idx = scores[keep].topk(self.max_proposals)
                keep = keep[idx]
            boxlist = boxlist[keep]
        return boxlist

def create_nms_func(nms_policy, score_thresh=0.,
        score_field='scores', max_proposals=-1):
    if nms_policy.TYPE == 'nms':
        return BoxListNMS(
            nms_policy.THRESH,
            max_proposals=max_proposals,
            score_field=score_field)
    elif nms_policy.TYPE == 'softnms':
        return BoxListSoftNMS(nms_policy.THRESH,
                max_proposals=max_proposals,
                score_field=score_field,
                score_thresh=score_thresh)
    elif nms_policy.TYPE == 'hnms':
        return BoxListHNMS(nms_policy, max_proposals, score_field)
    elif nms_policy.TYPE == 'compose_hnms_pt':
        return BoxListComposeHNMS(nms_policy, max_proposals,
                score_field)
