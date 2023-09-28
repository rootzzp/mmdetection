# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import copy
import warnings
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, is_norm
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList)
from ..task_modules.samplers import PseudoSampler
from ..utils import filter_scores_and_topk, images_to_levels, multi_apply
from .base_dense_head import BaseDenseHead
from mmengine.model import BaseModule
from typing import List, Optional, Sequence, Tuple, Union
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.logging import print_log
import math

def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor

def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x




@MODELS.register_module()
class YOLOv5Head(BaseDenseHead):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (Sequence[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (:obj:`ConfigDict` or dict): Config dict for anchor
            generator.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bounding box coder.
        featmap_strides (Sequence[int]): The stride of each scale.
            Should be in descending order. Defaults to (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Defaults to 0.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_conf (:obj:`ConfigDict` or dict): Config of confidence loss.
        loss_xy (:obj:`ConfigDict` or dict): Config of xy coordinate loss.
        loss_wh (:obj:`ConfigDict` or dict): Config of wh coordinate loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            YOLOV3 head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            YOLOV3 head. Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 anchor_generator: ConfigType = dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder: ConfigType = dict(type='YOLOBBoxCoder'),
                 one_hot_smoother: float = 0.,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: ConfigType = dict(
                     type='LeakyReLU', negative_slope=0.1),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xywh',
                     eps=1e-7,
                     reduction='mean',
                     loss_weight=0.05,
                     return_iou=True),
                 loss_obj: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 prior_match_thr: float = 4.0,
                 near_neighbor_thr: float = 0.5,
                 ignore_iof_thr: float = -1.0,
                 obj_level_weights: List[float] = [4.0, 1.0, 0.4],
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=None)
        # Check params

        self.head_module = MODELS.build(head_module)
        self.num_classes = self.head_module.num_classes
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)

        self.in_channels = head_module.in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], context=self)
            else:
                self.sampler = PseudoSampler()

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = TASK_UTILS.build(bbox_coder)

        self.prior_generator = TASK_UTILS.build(anchor_generator)

        self.loss_cls: nn.Module = MODELS.build(loss_cls)
        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)
        self.loss_obj: nn.Module = MODELS.build(loss_obj)

        self.num_levels = len(self.featmap_strides)
        self.featmap_sizes = [torch.empty(1)] * self.num_levels

        self.prior_match_thr = prior_match_thr
        self.near_neighbor_thr = near_neighbor_thr
        self.obj_level_weights = obj_level_weights
        self.ignore_iof_thr = ignore_iof_thr

        self.num_base_priors = self.prior_generator.num_base_priors[0]
        assert len(
            self.prior_generator.num_base_priors) == len(self.featmap_strides)
        self.special_init()

    @property
    def num_attrib(self) -> int:
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        assert len(self.obj_level_weights) == len(
            self.featmap_strides) == self.num_levels
        if self.prior_match_thr != 4.0:
            print_log(
                "!!!Now, you've changed the prior_match_thr "
                'parameter to something other than 4.0. Please make sure '
                'that you have modified both the regression formula in '
                'bbox_coder and before loss_box computation, '
                'otherwise the accuracy may be degraded!!!')

        if self.num_classes == 1:
            print_log('!!!You are using `YOLOv5Head` with num_classes == 1.'
                      ' The loss_cls will be 0. This is a normal phenomenon.')

        priors_base_sizes = torch.tensor(
            self.prior_generator.base_sizes, dtype=torch.float)
        featmap_strides = torch.tensor(
            self.featmap_strides, dtype=torch.float)[:, None, None]
        self.register_buffer(
            'priors_base_sizes',
            priors_base_sizes / featmap_strides,
            persistent=False)

        grid_offset = torch.tensor([
            [0, 0],  # center
            [1, 0],  # left
            [0, 1],  # up
            [-1, 0],  # right
            [0, -1],  # bottom
        ]).float()
        self.register_buffer(
            'grid_offset', grid_offset[:, None], persistent=False)

        prior_inds = torch.arange(self.num_base_priors).float().view(
            self.num_base_priors, 1)
        self.register_buffer('prior_inds', prior_inds, persistent=False)

    def forward(self, x: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        out = self.head_module(x)
        return tuple(out)

    def predict_by_feat(self,
                        pred_maps: Sequence[Tensor],
                        batch_img_metas: Optional[List[dict]],
                        cfg: OptConfigType = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results. It has been accelerated since PR #5991.

        Args:
            pred_maps (Sequence[Tensor]): Raw predictions for a batch of
                images.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (:obj:`ConfigDict` or dict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(pred_maps) == self.num_levels
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        num_imgs = len(batch_img_metas)
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]

        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=pred_maps[0].device)
        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_attrib)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(
                pred.new_tensor(stride).expand(pred.size(1)))

        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
        flatten_anchors = torch.cat(mlvl_anchors)
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = self.bbox_coder.decode(flatten_anchors,
                                                flatten_bbox_preds,
                                                flatten_strides.unsqueeze(-1))
        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            if conf_thr > 0:
                conf_inds = objectness >= conf_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            score_thr = cfg.get('score_thr', 0)
            nms_pre = cfg.get('nms_pre', -1)
            scores, labels, keep_idxs, _ = filter_scores_and_topk(
                scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                score_factors=objectness[keep_idxs],
            )
            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms,
                img_meta=img_meta)
            results_list.append(results)
        return results_list

    def loss_by_feat(
            self,
            pred_maps: Sequence[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        
        # 1. Convert gt to norm format
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        device = pred_maps[0].device
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        scaled_factor = torch.ones(7, device=device)

        for i in range(self.num_levels):
            batch_size, _, h, w = pred_maps[i].shape
            pred_map = pred_maps[i].view(batch_size, self.num_base_priors, self.num_attrib, h, w)
            cls_scores = pred_map[:, :, 5:, ...].reshape(batch_size, -1, h, w)
            bbox_preds = pred_map[:, :, :4, ...].reshape(batch_size, -1, h, w)
            objectnesses = pred_map[:, :, 4:5, ...].reshape(batch_size, -1, h, w)
            batch_size, _, h, w = bbox_preds.shape
            target_obj = torch.zeros_like(objectnesses)

            # empty gt bboxes
            if batch_targets_normed.shape[1] == 0:
                loss_box += bbox_preds.sum() * 0
                loss_cls += cls_scores.sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses, target_obj) * self.obj_level_weights[i]
                continue

            priors_base_sizes_i = self.priors_base_sizes[i]
            # feature map scale whwh
            scaled_factor[2:6] = torch.tensor(
                bbox_preds.shape)[[3, 2, 3, 2]]
            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_base_priors, num_bboxes, 7)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            # 2. Shape match
            wh_ratio = batch_targets_scaled[...,
                                            4:6] / priors_base_sizes_i[:, None]
            match_inds = torch.max(
                wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[match_inds]

            # no gt bbox matches anchor
            if batch_targets_scaled.shape[0] == 0:
                loss_box += bbox_preds.sum() * 0
                loss_cls += cls_scores.sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses, target_obj) * self.obj_level_weights[i]
                continue

            # 3. Positive samples with additional neighbors

            # check the left, up, right, bottom sides of the
            # targets grid, and determine whether assigned
            # them as positive samples as well.
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < self.near_neighbor_thr) &
                        (batch_targets_cxcy > 1)).T
            right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) &
                             (grid_xy > 1)).T
            offset_inds = torch.stack(
                (torch.ones_like(left), left, up, right, bottom))

            batch_targets_scaled = batch_targets_scaled.repeat(
                (5, 1, 1))[offset_inds]
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1],
                                                       1)[offset_inds]

            # prepare pred results and positive sample indexes to
            # calculate class loss and bbox lo
            _chunk_targets = batch_targets_scaled.chunk(4, 1)
            img_class_inds, grid_xy, grid_wh, priors_inds = _chunk_targets
            priors_inds, (img_inds, class_inds) = priors_inds.long().view(
                -1), img_class_inds.long().T

            grid_xy_long = (grid_xy -
                            retained_offsets * self.near_neighbor_thr).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T
            bboxes_targets = torch.cat((grid_xy - grid_xy_long, grid_wh), 1)

            # 4. Calculate loss
            # bbox loss
            retained_bbox_pred = bbox_preds.reshape(
                batch_size, self.num_base_priors, -1, h,
                w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
            priors_base_sizes_i = priors_base_sizes_i[priors_inds]
            decoded_bbox_pred = self._decode_bbox_to_xywh(
                retained_bbox_pred, priors_base_sizes_i)
            loss_box_i, iou = self.loss_bbox(decoded_bbox_pred, bboxes_targets)
            loss_box += loss_box_i

            # obj loss
            iou = iou.detach().clamp(0)
            target_obj[img_inds, priors_inds, grid_y_inds,
                       grid_x_inds] = iou.type(target_obj.dtype)
            loss_obj += self.loss_obj(objectnesses,
                                      target_obj) * self.obj_level_weights[i]

            # cls loss
            if self.num_classes > 1:
                pred_cls_scores = cls_scores.reshape(
                    batch_size, self.num_base_priors, -1, h,
                    w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

                target_class = torch.full_like(pred_cls_scores, 0.)
                target_class[range(batch_targets_scaled.shape[0]),
                             class_inds] = 1.
                loss_cls += self.loss_cls(pred_cls_scores, target_class)
            else:
                loss_cls += cls_scores.sum() * 0

        return dict(
            loss_cls=loss_cls * batch_size ,
            loss_obj=loss_obj * batch_size ,
            loss_bbox=loss_box * batch_size)
    
    def _decode_bbox_to_xywh(self, bbox_pred, priors_base_sizes) -> Tensor:
        bbox_pred = bbox_pred.sigmoid()
        pred_xy = bbox_pred[:, :2] * 2 - 0.5
        pred_wh = (bbox_pred[:, 2:] * 2)**2 * priors_base_sizes
        decoded_bbox_pred = torch.cat((pred_xy, pred_wh), dim=-1)
        return decoded_bbox_pred
    
    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):
            # fast version
            img_shape = batch_img_metas[0]['batch_input_shape']
            gt_bboxes_xyxy = batch_gt_instances[:, 2:]
            xy1, xy2 = gt_bboxes_xyxy.split((2, 2), dim=-1)
            gt_bboxes_xywh = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
            gt_bboxes_xywh[:, 1::2] /= img_shape[0]
            gt_bboxes_xywh[:, 0::2] /= img_shape[1]
            batch_gt_instances[:, 2:] = gt_bboxes_xywh

            # (num_base_priors, num_bboxes, 6)
            batch_targets_normed = batch_gt_instances.repeat(
                self.num_base_priors, 1, 1)
        else:
            batch_target_list = []
            # Convert xyxy bbox to yolo format.
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]['batch_input_shape']
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels

                xy1, xy2 = bboxes.split((2, 2), dim=-1)
                bboxes = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
                # normalized to 0-1
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0::2] /= img_shape[1]

                index = bboxes.new_full((len(bboxes), 1), i)
                # (batch_idx, label, normed_bbox)
                target = torch.cat((index, labels[:, None].float(), bboxes),
                                   dim=1)
                batch_target_list.append(target)

            # (num_base_priors, num_bboxes, 6)
            batch_targets_normed = torch.cat(
                batch_target_list, dim=0).repeat(self.num_base_priors, 1, 1)

        # (num_base_priors, num_bboxes, 1)
        batch_targets_prior_inds = self.prior_inds.repeat(
            1, batch_targets_normed.shape[1])[..., None]
        # (num_base_priors, num_bboxes, 7)
        # (img_ind, labels, bbox_cx, bbox_cy, bbox_w, bbox_h, prior_ind)
        batch_targets_normed = torch.cat(
            (batch_targets_normed, batch_targets_prior_inds), 2)
        return batch_targets_normed
    
@MODELS.register_module()
class YOLOv5HeadModule(BaseModule):
    """YOLOv5Head head module used in `YOLOv5`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to (8, 16, 32).
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 3,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.widen_factor = widen_factor

        self.featmap_strides = featmap_strides
        self.num_out_attrib = 5 + self.num_classes
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors

        if isinstance(in_channels, int):
            self.in_channels = [make_divisible(in_channels, widen_factor)
                                ] * self.num_levels
        else:
            self.in_channels = [
                make_divisible(i, widen_factor) for i in in_channels
            ]

        self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv5 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_base_priors * self.num_out_attrib,
                                  1)

            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv5 head."""
        super().init_weights()
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            b = mi.bias.data.view(self.num_base_priors, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s)**2)
            # NOTE: The following initialization can only be performed on the
            # bias of the category, if the following initialization is
            # performed on the bias of mask coefficient,
            # there will be a significant decrease in mask AP.
            b.data[:, 5:5 + self.num_classes] += math.log(
                0.6 / (self.num_classes - 0.999999))

            mi.bias.data = b.view(-1)

    def forward(self, x) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        assert len(x) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            pred_map = self.convs_pred[i](x[i])
            pred_maps.append(pred_map)
        return tuple(pred_maps),

    def forward_single(self, x: Tensor,
                       convs: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""

        pred_map = convs(x)
        return pred_map
