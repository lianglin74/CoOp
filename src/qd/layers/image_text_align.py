from qd.layers.tensor_queue import TensorQueue
from qd.torch_common import describe_tensor
import torch
import logging
from torch import nn
from qd.qd_common import print_frame_info


class ImageTextContrastive(nn.Module):
    def __init__(self,
                 normalize=False,
                 temperature=0.2,
                 abstract_image_type='avg',
                 abstract_text_type='avg',
                 loss_style='batch',
                 queue_size=1024,
                 feat_dim=384,
                 sinkhorn_eps=0.05,
                 batch_sink_weight=1.,
                 batch_weight_sink=1.,
                 ):
        super().__init__()
        print_frame_info()
        self.normalize = normalize
        from qd.layers.loss import MultiHotCrossEntropyLoss
        self.loss = MultiHotCrossEntropyLoss()
        self.iter = 0
        self.temperature = temperature

        assert abstract_image_type in ['avg', 'first', 'I']
        self.abstract_image_type = abstract_image_type
        assert abstract_text_type in ['avg', 'first', 'last', 'I']
        self.abstract_text_type = abstract_text_type
        self.loss_style = loss_style
        self.batch_weight_sink = batch_weight_sink
        self.batch_sink_weight = batch_sink_weight

        if self.loss_style == 'moco':
            self.queue_im_feats = TensorQueue(queue_size, feat_dim)
            self.queue_im_hash = TensorQueue(queue_size, 1, normalize=False)
            self.queue_text_feats = TensorQueue(queue_size, feat_dim)
            self.queue_text_hash = TensorQueue(queue_size, 1, normalize=False)
        elif loss_style in ['sinkhorn', 'batch_sink']:
            from qd.layers.loss import SinkhornClusterLoss
            # we re-run queue_size parameter here
            self.sinkhorn_loss = SinkhornClusterLoss(
                queue_size,
                feat_dim,
                eps=sinkhorn_eps,
                T=temperature,
            )

    def abstract_text(self, text_info):
        if self.abstract_text_type == 'avg':
            valid_mask = text_info['input_ids'] != 0
            valid_token = valid_mask.sum(dim=1)[:, None]
            assert (valid_token >= 1).sum() == valid_token.numel()
            # each input has at least [CLS] token and thus each input has at least
            # one valid token
            text_feats = text_info['text_feats']
            text_feats = (text_feats * valid_mask[:, :, None]).sum(dim=1)
            text_feats = text_feats / valid_token
        elif self.abstract_text_type == 'first':
            text_feats = text_info['text_feats']
            text_feats = text_feats[:, 0, :].clone()
        elif self.abstract_text_type == 'last':
            input_ids = text_info['input_ids']
            num = input_ids.shape[0]
            aug_input_ids = torch.cat((
                input_ids, torch.zeros((num, 1), device=input_ids.device)
            ), dim=1)
            idx = (aug_input_ids == 0).int().argmax(dim=1) - 1

            text_feats = text_info['text_feats']
            text_feats = text_feats[torch.arange(len(text_feats)), idx]
        else:
            text_feats = text_info['text_feats']
        if self.normalize:
            text_feats = nn.functional.normalize(text_feats, dim=1)
        return text_feats

    def abstract_image(self, img_feats):
        if self.abstract_image_type == 'avg':
            img_feats = img_feats.mean(dim=1)
        elif self.abstract_image_type == 'first':
            img_feats = img_feats[:, 0, :].clone()
        if self.normalize:
            img_feats = nn.functional.normalize(img_feats, dim=1)
        return img_feats

    def forward(self, img_info, text_info):
        img_feats = img_info['img_feats']
        verbose = (self.iter % 100) == 0
        self.iter += 1
        #ipdb> pp img_feats.shape
        #torch.Size([4, 49, 384])
        #ipdb> text_feats.shape
        #torch.Size([4, 17, 384])
        if verbose:
            logging.info('image features = {}'.format(describe_tensor(img_feats)))

        img_feats = self.abstract_image(img_feats)
        # we assume 0 is pad token
        text_feats = self.abstract_text(text_info)

        if verbose:
            with torch.no_grad():
                x = torch.matmul(img_feats, img_feats.t())
                logging.info('pair wise image difference = {}'.format(
                    describe_tensor(x)))

        from qd.torch_common import hash_tensor_prime_simple
        text_hash = hash_tensor_prime_simple(text_info['origin_input_ids'])

        if self.loss_style == 'batch':
            loss = self.loss_batch_style(
                img_feats, text_feats, text_hash,
                img_info['idx_img'], verbose)
        elif self.loss_style == 'moco':
            loss = self.loss_moco_style(
                img_feats, text_feats, text_hash,
                img_info['idx_img'], verbose,
            )
        elif self.loss_style == 'sinkhorn':
            loss = self.sinkhorn_loss(
                img_feats, text_feats,
            )
        elif self.loss_style == 'batch_sink':
            loss_batch = self.loss_batch_style(
                img_feats, text_feats, text_hash,
                img_info['idx_img'], verbose)
            loss_sink = self.sinkhorn_loss(
                img_feats, text_feats,
            )
            loss = {
                'loss_batch': loss_batch * self.batch_weight_sink,
                'loss_sink': loss_sink * self.batch_sink_weight,
            }
        else:
            raise NotImplementedError(self.loss_style)

        return loss

    def loss_moco_style(self, img_feats, text_feats, text_hash, image_hash,
                        verbose):
        l1 = self.loss_moco_left_to_right(
            img_feats,
            text_feats,
            self.queue_text_feats.queue,
            image_hash,
            text_hash,
            self.queue_im_hash.queue,
            self.queue_text_hash.queue,
        )
        l2 = self.loss_moco_left_to_right(
            text_feats,
            img_feats,
            self.queue_im_feats.queue,
            text_hash,
            image_hash,
            self.queue_text_hash.queue,
            self.queue_im_hash.queue,
        )

        from qd.torch_common import concat_all_gather
        img_feats = concat_all_gather(img_feats)
        image_hash = concat_all_gather(image_hash)
        text_feats = concat_all_gather(text_feats)
        text_hash = concat_all_gather(text_hash)

        self.queue_im_feats.en_de_queue(img_feats)
        self.queue_im_hash.en_de_queue(image_hash.reshape((-1, 1)))
        self.queue_text_feats.en_de_queue(text_feats)
        self.queue_text_hash.en_de_queue(text_hash.reshape((-1, 1)))

        return (l1 + l2) / 2.

    def loss_moco_left_to_right(self, xs, ys, queue_ys,
                                xs_hash,
                                ys_hash,
                                queue_xs_hash,
                                queue_ys_hash):
        pos = (xs * ys).sum(dim=1, keepdim=True)
        neg = torch.matmul(xs, queue_ys.clone().t())
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature

        xs_hash = xs_hash.reshape((-1, 1))
        queue_xs_hash = queue_xs_hash.reshape((1, -1)).expand(len(xs_hash), -1)
        all_xs_hash = torch.cat((xs_hash, queue_xs_hash), dim=1)
        gt1 = xs_hash.reshape((-1, 1)) == all_xs_hash

        ys_hash = ys_hash.reshape((-1, 1))
        queue_ys_hash = queue_ys_hash.reshape((1, -1)).expand(len(ys_hash), -1)
        all_ys_hash = torch.cat((ys_hash, queue_ys_hash), dim=1)
        gt2 = ys_hash.reshape((-1, 1)) == all_ys_hash

        gt = torch.logical_or(gt1, gt2).float()

        l = self.loss(logits, gt)
        return l

    def loss_batch_style(self, img_feats, text_feats, text_hash, image_hash,
                         verbose):
        from qd.torch_common import all_gather_grad_curr
        all_img_feats = all_gather_grad_curr(img_feats)
        all_text_feats = all_gather_grad_curr(text_feats)
        all_text_hash = all_gather_grad_curr(text_hash)
        all_image_hash = all_gather_grad_curr(image_hash)

        gt_text = all_text_hash.reshape((-1, 1)) == all_text_hash.reshape((1, -1))
        gt_image = all_image_hash.reshape((-1, 1)) == all_image_hash.reshape((1, -1))
        gt = torch.logical_or(gt_text, gt_image)
        if verbose:
            total_num_pos = gt.sum()
            e = torch.eye(gt_image.shape[0]).to(gt_image.device)
            off_pos = torch.logical_and(gt_image, torch.logical_not(e)).sum()
            logging.info('off_pos/total_pos = {}/{}'.format(off_pos, total_num_pos))

        gt = gt.float()
        logits = torch.matmul(all_img_feats, all_text_feats.t())
        logits /= self.temperature
        if verbose:
            logging.info('logits = {}'.format(describe_tensor(logits)))
        l1 = self.loss(logits, gt)
        l2 = self.loss(logits.t(), gt.t())
        l = (l1 + l2) / 2.

        from qd.qd_common import get_mpi_size
        # DistributedDataParallel will do average on gradients. Thus, here, we
        # have to multiply it with the world size
        return l * get_mpi_size()

def create_align_loss(align_loss_type, temperature=0.2, **kwargs):
    loss = None
    if align_loss_type.endswith('_n'):
        normalize = True
        align_loss_type = align_loss_type[:-2]
    else:
        normalize = False
    image_t, text_t = align_loss_type.split('_')
    loss = ImageTextContrastive(
        abstract_image_type=image_t,
        abstract_text_type=text_t,
        normalize=normalize,
        temperature=temperature,
        **kwargs,
    )
    return loss

class ImageTextAligner(nn.Module):
    def __init__(self,
                 image_encoder,
                 text_encoder,
                 align_loss,
                 temperature=0.2,
                 **kwargs,
                 ):
        super().__init__()
        print_frame_info()
        from qd.qd_common import execute_func
        if isinstance(image_encoder, dict):
            self.image_encoder = execute_func(image_encoder)
        else:
            self.image_encoder = image_encoder

        if isinstance(text_encoder, dict):
            self.text_encoder = execute_func(text_encoder)
        else:
            self.text_encoder = text_encoder

        self.align_loss = create_align_loss(
            align_loss,
            temperature,
            **kwargs,
        )

    def feed_test_texts(self, data):
        position_ids = None
        if 'position_ids' in data:
            position_ids = data['position_ids'].unsqueeze(0)
        token_type_ids = None
        if 'token_type_ids' in data:
            token_type_ids = data['token_type_ids'].unsqueeze(0)
        attention_mask = None
        if 'attention_mask' in data:
            attention_mask = data['attention_mask'].unsqueeze(0)
        input_ids = data['input_ids'].unsqueeze(0)
        text_feats = self.text_encoder(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        text_info = {
            'text_feats': text_feats['pooled_output'],
            'input_ids': input_ids,
        }
        x = self.align_loss.abstract_text(text_info)
        return x

    def forward(self, data):
        img_feats = self.image_encoder(data['image'])
        if self.training:
            loss_dict = {}
            text_feats = self.text_encoder(
                data['input_ids'],
                position_ids=data.get('position_ids'),
                token_type_ids=data.get('token_type_ids'),
                attention_mask=data.get('attention_mask'),
            )
            align_loss = self.align_loss(
                {
                    'img_feats': img_feats,
                    'idx_img': data['idx_img'],
                },
                {
                    'text_feats': text_feats['pooled_output'],
                    'input_ids': data['input_ids'],
                    'origin_input_ids': data['origin_input_ids'],
                })

            if isinstance(align_loss, dict):
                loss_dict.update(align_loss)
            else:
                loss_dict['align_loss'] = align_loss

            return loss_dict
        else:
            img_feats = self.align_loss.abstract_image(img_feats)
            result = torch.matmul(img_feats, self.all_abstracted_text.t())
            result /= self.align_loss.temperature
            result = torch.nn.functional.softmax(result, dim=1)
            return result

