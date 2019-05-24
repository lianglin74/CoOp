import torch
import time
import torch.nn as nn
from ..utils.accuracy import get_accuracy_calculator
from ..utils.averagemeter import AverageMeter


def validate(val_loader, model, criterion, logger):
    batch_time = AverageMeter()
    accuracy = get_accuracy_calculator(multi_label=not isinstance(criterion, nn.CrossEntropyLoss))
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        val_loader_len = len(val_loader)
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            all_outputs = model(input)
            if isinstance(all_outputs, tuple):
                output, feature = all_outputs[0], all_outputs[1]
            else:
                output = all_outputs
            loss = criterion(output, target)

            # measure accuracy and record loss
            accuracy.calc(output, target)
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0 or i == val_loader_len - 1:
                info_str = 'Test: [{0}/{1}]\tLoss {loss.avg:.4f}\t'.format(i, val_loader_len, loss=losses)
                info_str += accuracy.result_str()
                logger.info(info_str)

    return accuracy.prec()
