import argparse
import os
import sys
import torch 
import shutil
import utils
import numpy as np
import metrics
import torch.nn as nn
import torch.backends.cudnn as cudnn 
import vision_transformer as vits

from tqdm import tqdm
from datasets import build_dataset
from meters import AverageEpochMeter
from object_discovery import ncut, detect_box, get_feats
from visualization import visualize_fms, visualize_predictions_gt, visualize_img, visualize_eigvec

from collections import OrderedDict
#from visualizations import visualize_img, visualize_eigvec, visualize_gt, visualize_fms, visualize_predictions, visualize_predictions_gt

def evaluate():
    if args.device != 'cuda':
        args.distributed = False
    else:
        utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True
    
    # =============build network ==============
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=args.num_labels)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
        print(f'embed_dim: {embed_dim}')
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
  
    model.to(device)
    model.eval()

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.to(device)

    # ============ Load checkpoint ============= 
    checkpoint_model = torch.load(args.pretrained_weights, map_location='cpu')
    if args.classifier_weights is None:
        linear_classifier.load_state_dict(checkpoint_model['classifier'])
        model.load_state_dict(checkpoint_model['model'], strict=False)
    else:
        model.load_state_dict(checkpoint_model)
        checkpoint_classifier = torch.load(args.classifier_weights, map_location='cpu')
        new_state_dict = OrderedDict()
        new_state_dict = {k[7:]: v for k, v in checkpoint_classifier['state_dict'].items()}
        # load params
        linear_classifier.load_state_dict(new_state_dict)
    print('Load from checkpoint done.')
   
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
     
    num_batches = len(val_loader)
    scales = [args.patch_size, args.patch_size]
    top1_cls_meter = AverageEpochMeter('Top-1 Cls')
    top5_cls_meter = AverageEpochMeter('Top-5 Cls')
    gt_loc_meter = AverageEpochMeter('GT-Known Loc')
    top1_loc_meter = AverageEpochMeter('Top-1 Loc')
    feat_out = {}
    bbox_error = 0
    cls_error = 0
    skip = 0 
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
    val_loader = tqdm(val_loader)
    for i, (images, labels, gt_boxes) in enumerate(val_loader):
        init_image_size = images[0].shape
        images = utils.padding_img(images[0], args)
        images = images.unsqueeze(0)
        if images.shape[-1] > 1000 and images.shape[-2] > 1000:
            skip = skip + 1
            continue
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            
            intermediate_output, shape= model.get_intermediate_layers(images, args.n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if args.avgpool_patchtokens:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
            output = linear_classifier(output)
            
            batch_size = images.size(0)
            top1_cls, top5_cls = utils.accuracy(output, labels, topk=(1, 5))
            top1_cls_meter.update(top1_cls, batch_size)
            top5_cls_meter.update(top5_cls, batch_size)



        # ===== Generate Bbox=========
        dims = (images[0].shape[-2] // args.patch_size, images[0].shape[-1] // args.patch_size)
        k = get_feats(feat_out, shape)
        bboxes, mask, seed, eigvec= ncut(k, dims, scales, init_image_size=init_image_size[1:], eps=args.eps, tau=args.tau)
        bboxes_copy = np.copy(bboxes)
        bboxes = torch.FloatTensor(bboxes).unsqueeze(0)
        gt_loc, top1_loc = metrics.loc_accuracy(output, labels, gt_boxes, bboxes)
       
        if args.visual:
            visualize_predictions_gt(images[0], bboxes_copy , gt_boxes, i, seed, dims, scales, './output/all')
            visualize_eigvec(eigvec, './output/all', i, dims, scales)

        #visualize_fms(images[0], mask, seed, i, dims, scales) 
        if top1_loc == 0 and gt_loc == 0: 
            bbox_error = bbox_error + 1
            if args.visual:
                #  print(f'i: {i}, gt_boxes {gt_boxes}, prediction: {bboxes_copy}, image size: {images.shape}')
                visualize_predictions_gt(images[0], bboxes_copy , gt_boxes, i, seed, dims, scales, './output/bbox_error')
            
        elif top1_loc == 0 and gt_loc == 1:
            cls_error = cls_error +1
            if args.visual:
                visualize_predictions_gt(images[0], bboxes_copy , gt_boxes, i, seed, dims, scales, './output/classification_error')

        gt_loc_meter.update(gt_loc, batch_size)
        top1_loc_meter.update(top1_loc, batch_size)
        top1_cls = top1_cls_meter.compute()
        top5_cls = top5_cls_meter.compute()
        gt_loc = gt_loc_meter.compute()
        top1_loc = top1_loc_meter.compute()
        val_loader.set_description(f'Top1_cls: {top1_cls:.4f}, top5_cls{top5_cls:.4f}, gt_loc: {gt_loc:.4f}, top1_loc:{top1_loc:.4f}')
    print(f'Top1_cls: {top1_cls}, top5_cls{top5_cls}, gt_loc: {gt_loc}, top1_loc:{top1_loc}')
    print(f'Bbox error: {bbox_error}, cls error: {cls_error}')
    print(f'Skip large image: {skip}')
       
           
class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--dataset', default='cub', type=str, choices=['cub', 'imagenet'], help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size, default(224).')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--classifier_weights', default=None, type=str, help="Path to linear classifier pretrained weights to evaluate.")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--num_labels', default=200, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--distributed', default=False, action='store_true', help='device to use for training / testing')
    parser.add_argument('--ori_size', default=False, action='store_true', help='Evaluate on image raw size')
    parser.add_argument('--visual', default=False, action='store_true', help='Visualize error examples on ./test')
    parser.add_argument('--eps', default=1e-5, type=float, help='hyperparameter for tokencut')
    parser.add_argument('--tau', default=0.05, type=float, help='hyperparamter for tokencut')
    parser.add_argument('--no_center_crop', default=False, action='store_true', help='Center crop input image')
    args = parser.parse_args()
    evaluate()
