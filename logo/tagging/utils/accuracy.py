import torch
from ..utils.averagemeter import AverageMeter


class Accuracy(object):
    """ base class for accuracy calculation
    """
    def __init__(self):
        pass

    def calc(self, output, target):
        pass

    def prec(self):
        pass

    def result_str(self):
        pass

class SingleLabelAccuracy(Accuracy):
    """ class for single label accuracy calculation
    """
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.topk_acc = [AverageMeter() for _ in topk]

    def calc(self, output, target):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for i,k in enumerate(self.topk):
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                acc = correct_k.mul_(100.0 / batch_size).item()
                self.topk_acc[i].update(acc, output.size(0))

    def prec(self):
        return self.topk_acc[0].avg

    def result_str(self):
        acc_str = ['Prec@{k} {acc.val:.3f} ({acc.avg:.3f})'.format(k=self.topk[i], acc=self.topk_acc[i]) \
                    for i in range(len(self.topk))]
        return '\t'.join(acc_str)


class MultiLabelAccuracy(Accuracy):
    """ class for multi label accuracy calculation
    """
    def __init__(self):
        self.accuracy = AverageMeter()

    def calc(self, output, target):
        """Computes the precision of multi label prediction"""
        with torch.no_grad():
            batch_size = target.size(0)
            num_classes = target.size(1)

            num_labels = target.sum(dim=1)
            maxk = num_labels.max().int().item()
            _, pred_topk = output.topk(maxk, dim=1, largest=True)
            n = 0
            accuracy = 0.
            for i in range(batch_size):
                k = num_labels[i].int().item()
                if k == 0:
                    continue
                n += 1
                pred = torch.ones(num_classes).cuda().mul(-1)
                pred[pred_topk[i,:k]] = 1
                correct = pred.eq(target[i])
                accuracy += correct.float().sum() * 100. / k

            assert n > 0, 'Expect at least one positive label in a batch'
            accuracy /= n
            self.accuracy.update(accuracy, output.size(0))

    def prec(self):
        return self.accuracy.avg

    def result_str(self):
        return 'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(acc=self.accuracy)

def get_accuracy_calculator(multi_label=False):
    if multi_label:
        acc = MultiLabelAccuracy()
    else:
        acc = SingleLabelAccuracy(topk=(1, 5))
    return acc
