import torch
import time
import torch.nn as nn
from ..utils.accuracy import get_accuracy_calculator
from ..utils.averagemeter import AverageMeter


def validate(val_loader, model, criterion, logger):
    batch_time = AverageMeter()
    accuracy = get_accuracy_calculator(multi_label=not isinstance(criterion, nn.CrossEntropyLoss))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        val_loader_len = len(val_loader)
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            all_outputs = model(input)
            output, feature = all_outputs[0], all_outputs[1]

            # measure accuracy and record loss
            accuracy.calc(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0 or i == val_loader_len - 1:
                info_str = 'Test: [{0}/{1}]\t'.format(i, val_loader_len)
                info_str += accuracy.result_str()
                logger.info(info_str)

    return accuracy.prec()
