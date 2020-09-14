import torch
from tqdm import tqdm
from qd.tsv_io import tsv_reader
import json
from qd.data_layer.loader import create_feature_loader
from qd.tsv_io import TSVDataset
import logging
from qd.qd_common import json_dump
from qd.qd_common import get_mpi_size, get_mpi_rank
from qd.tsv_io import tsv_writer
from qd.process_tsv import concat_tsv_files
from qd.process_tsv import delete_tsv_files
from qd.torch_common import synchronize
import os.path as op


def knn_classifier(data, split, version, feature, k, test_feature,
                   pred_out=None):
    assert version == 0
    if pred_out is None:
        pred_out = test_feature + '.knn{}.pred.tsv'.format(k)
    if op.isfile(pred_out):
        logging.info('{} exists'.format(pred_out))
        return pred_out
    from qd.qd_pytorch import ensure_init_process_group
    ensure_init_process_group()

    test_loader = create_feature_loader(test_feature)

    keys = torch.tensor([json.loads(str_rects)[0]['feature'] for _, str_rects in
                         tqdm(tsv_reader(feature))])
    while keys.shape[-1]== 1:
        keys = keys.squeeze(dim=-1)

    import torch.nn.functional as F
    keys = F.normalize(keys, dim=1)
    dataset = TSVDataset(data)
    #labelmap = dataset.load_labelmap()
    #labels = [json.loads(str_rects)[0]['class'] for key, str_rects in
              #dataset.iter_data(split, 'label', version=version)]
    logging.info('loading labels')
    num_unique_labels = len(dataset.load_labelmap())
    labels = [int(str_rects) for key, str_rects in
              dataset.iter_data(split, 'label', version=version)]
    labels = torch.tensor(labels)
    def gen_rows():
        for info in tqdm(test_loader):
            query = info['feature']
            query = query.squeeze(dim=1)
            query = F.normalize(query)
            similarity = torch.matmul(query, keys.T)
            if k == 1:
                score, idx = similarity.max(dim=1)
                label = labels[idx]
                for key, l, s in zip(info['key'], label, score):
                    yield key, json_dump([{'class': str(int(l)), 'conf':
                                           float(s)}])
            else:
                raise ValueError('not ready for multi image')
                score, idx = torch.topk(similarity, dim=1, k=k)
                score = (score / 0.07).exp()
                full_score = torch.zeros((len(query), k, num_unique_labels))
                topk_labels = labels[idx]
                full_score.scatter_(2, topk_labels[:, None], score[:, None])
                pred = full_score.sum(dim=1)
                score, idx = pred.topk(k=10)
                score = score.tolist()
                idx = idx.tolist()
                yield key, json_dump([{'class': str(i), 'conf': float(s)}
                                  for i, s in zip(idx[0], score[0])])
    if get_mpi_size() == 1:
        curr_pred_out = pred_out
    else:
        curr_pred_out = pred_out + '.{}.{}.tsv'.format(
            get_mpi_rank(),
            get_mpi_size())
    tsv_writer(gen_rows(), curr_pred_out)
    if get_mpi_size() > 1:
        synchronize()
        if get_mpi_rank() == 0:
            cached = [pred_out + '.{}.{}.tsv'.format(
                i,
                get_mpi_size()) for i in range(0, get_mpi_size())]
            concat_tsv_files(cached, pred_out)
            delete_tsv_files(cached)

    return pred_out
