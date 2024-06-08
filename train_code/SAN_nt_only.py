import sys
from matplotlib import pyplot as plt
from numpy import mat
sys.path.append('/home/cqzeng/SAN')
import time
import os
from pyparsing import C, Optional, line
import torch
import torch.nn as nn
from timm.data import resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
import timm.loss
from timm.models.layers import _assert
from timm.scheduler import create_scheduler
from data import create_loader, create_dataset
from utils import create_optimizer_v2, optimizer_kwargs
# from optim_factory import create_optimizer_v2, optimizer_kwargs
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
from args import _parse_args
import wandb
def init_model_and_dataloader(args_input):
    
    def create_dataload(args, num_aug_splits:int = 0, data_config: resolve_data_config = None):
        print("8888888888888888888888888888888888 Dataset Info 88888888888888888888888888888888888888888")
        print('Data name: ', args.dataset)
        print('Data dir: ', args.data_dir)
        print('Class num: ', args.dataset.split('_')[-1])
        print('Train split: ', args.train_split)
        print('Val split: ', args.val_split)
        print('Image size: ', data_config['input_size'])
        print('Batch size: ', args.batch_size)
        print('Auto augment: ', args.aa)
        print('Color jitter: ', args.color_jitter)
        print('Random erasing: ', args.reprob)
        print('Random crop ratio: ', args.ratio)
        # setup mixup / cutmix
        collate_fn = None
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.num_classes if args.dataset.split('_')[-1] == args.num_classes else int(args.dataset.split('_')[-1]),)
            if args.prefetcher:
                assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                collate_fn = FastCollateMixup(**mixup_args) # type: ignore
                print('Mixup/Cutmix: FastCollateMixup, a type of prefetched fast mixup but use more memory')
            else:
                mixup_fn = Mixup(**mixup_args) # type: ignore
                print('Mixup/Cutmix: Normal no prefetched mixup but use less memory')
        else:
            print("Mixup/Cutmix: No mixup/cutmix")

        # wrap dataset in AugMix helper
        if num_aug_splits > 1:
            dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

        # create data loaders w/ augmentation pipeiine
        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            train_interpolation = data_config['interpolation']
        # create the train and eval datasets based on the model 
        dataset_train = create_dataset(
            args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats,
            )

        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size)
        
        
        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            simple_aug=args.simple_aug,
            contrast_aug=args.contrast_aug,
            direct_resize=args.direct_resize,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_repeats=args.aug_repeats,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            worker_seeding=args.worker_seeding,
        )

        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            direct_resize=args.direct_resize,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            worker_seeding=args.worker_seeding,
        )
        return  loader_train, loader_eval, mixup_active, mixup_fn
    
    args, args_text = _parse_args(args_input)
    if type(args.dataset) ==list and len(args.dataset)==1:
        args.dataset = args.dataset[0]
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False
    args.prefetcher = not args.no_prefetcher
    
    model = create_model(
    args.model,
    pretrained=args.pretrained,
    num_classes=args.num_classes,
    drop_rate=args.drop,
    drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
    drop_path_rate=args.drop_path,
    drop_block_rate=args.drop_block,
    global_pool=args.gp,
    scriptable=args.torchscript,
    checkpoint_path=args.initial_checkpoint,
    ).cuda()
    print("8888888888888888888888888888888888 Model Info 88888888888888888888888888888888888888888")
    print("Model name: ", args.model)
    print("Model: ", model)
    # print(f'model parameters count (neuron):  {sum([m.numel() for m in model.parameters()])}')


    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)
    
    data_config = resolve_data_config(vars(args), model=model,
                                    #   verbose=args.local_rank == 0
                                      )
    
    loader_train, loader_eval, mixup_active, mixup_fn = create_dataload(args=args,
                                                                        data_config=data_config)
    return model, loader_train, loader_eval, mixup_active, mixup_fn
    
def create_loss_fn(args):
    name = ''
    if args.mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = timm.loss.BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
            name = 'BCE loss'
        else:
            train_loss_fn = timm.loss.SoftTargetCrossEntropy()
            name = 'SoftTargetCrossEntropy loss'
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = timm.loss.BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
            name = 'BCE loss'       
        else:
            train_loss_fn = timm.loss.LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            name = 'LabelSmoothingCrossEntropy loss'
    else:
        train_loss_fn = nn.CrossEntropyLoss()
        name = 'CrossEntropy loss'
    train_loss_fn = train_loss_fn.cuda()
    return train_loss_fn, name

def train(model, loader, optimizer, loss_fn, lr_scheduler, args):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    total_mian_loss = 0
    total_aux_loss = 0
    total_batches = len(loader)
    number_updates = args.current_epoch*total_batches
    progress_bar = tqdm(enumerate(loader), total=total_batches, desc='Training', leave=True)
    for batch_idx, (inputs, targets) in progress_bar:
        optimizer.zero_grad()
        if args.mixup_fn is not None:
            inputs, targets = args.mixup_fn(inputs, targets)
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        main_loss = loss_fn(outputs, targets)
        aux_loss, counter = 0, 0
        for module in model.modules():
            if hasattr(module, 'mse_loss'):
                aux_loss += module.mse_loss
                counter += 1
        aux_loss = aux_loss / counter if counter != 0 else 0
        loss = main_loss
        loss.backward()
        optimizer.step()
        current_lr = optimizer.param_groups[0]['lr']
        total_loss += loss.item()
        total_mian_loss += main_loss.item()
        # total_aux_loss += aux_loss.item()
        progress_bar.set_postfix(loss=total_loss/(batch_idx+1),
                                 main_loss = total_mian_loss/(batch_idx+1),
                                #  aux_loss = total_aux_loss/(batch_idx+1),
                                 lr=current_lr)
        number_updates+=1
        lr_scheduler.step_update(num_updates=number_updates)
    args.current_epoch += 1
    return total_loss/(batch_idx+1), current_lr

def validate(model, loader, loss_fn, args):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct, total = 0, 0

    running_loss_head_only = 0.0
    correct_head_only, total_head_only = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Validation', leave=False):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            outputs_head_only = model.forward_head_only(inputs)
            loss_head_only = loss_fn(outputs_head_only, targets)
            running_loss_head_only += loss_head_only.item()
            _, predicted_head_only = torch.max(outputs_head_only.data, 1)
            total_head_only += targets.size(0)
            correct_head_only += (predicted_head_only == targets).sum().item()
            
    
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    
    avg_loss_head_only = running_loss_head_only / len(loader)
    accuracy_head_only = 100 * correct_head_only / total_head_only

    return avg_loss, accuracy, avg_loss_head_only, accuracy_head_only


class neurotransmitter(nn.Module):
    '''
    This class initialize a trainable tensor to simulate the effect of neurotransmitter.
    self.scale: trainable tensor
    return: x * self.scale
    '''
    def __init__(self, input_dim):
        super(neurotransmitter, self).__init__()
        self.scale = nn.Parameter(torch.ones(input_dim))
    def forward(self, x):
        return x * self.scale
    def __repr__(self):
        return f"{self.__class__.__name__}(scale shape: {self.scale.shape})"
    def visualize(self, save_path=None):
        scale = self.scale.detach().cpu().numpy()
        if scale.ndim == 1:
            scale = scale.reshape(1, -1)
        plt.figure(figsize=(10, 5 if scale.shape[0] == 1 else 10))
        plt.xlabel('Neuron index')
        plt.imshow(scale, cmap='hot', aspect='auto')
        plt.colorbar()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

# reusing
class LTD_LTP(neurotransmitter):
    '''
    This class initialize a trainable tensor to simulate the effect of neurotransmitter.
    '''
    pass

class synapse_plasticity(nn.Module):
    '''
    This class initialize neurotransmitter instance and apply it to simulate the effect of synapse plasticity.
    '''
    def __init__(self, weight, bias, presynapse: neurotransmitter = None): # type: ignore
        super(synapse_plasticity, self).__init__()
        self.weight = weight
        self.bias = bias
        self.weight_shape = weight.shape
        if self.weight.dim() == 2:
            self.func_type = nn.functional.linear
        elif self.weight.dim() == 4:
            self.func_type = nn.functional.conv2d
        self.presynapse = presynapse
        if self.presynapse:
            self.LTD_LTP = LTD_LTP(self.presynapse.scale.shape[0])
        self.postsynapse = neurotransmitter(self.weight_shape[0])
    
    def forward(self, x, nt_sp_both = 'both'):
        if nt_sp_both == 'nt':
            x = self.func_type(x, self.weight, self.bias)
            x = self.postsynapse(x)
            return x
        else: 
            weight = self.weight.clone()
            if self.presynapse is not None:
                scale = self.presynapse.scale
                scale = self.LTD_LTP(scale)
                weight *= scale
            x = self.func_type(x, weight, self.bias)
            if nt_sp_both == 'both':
                x = self.postsynapse(x)
            return x
    
    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(\n"
        repr_str += f"  Weight shape: {self.weight_shape}\n"
        repr_str += f"  Presynapse: {self.presynapse}\n"
        if self.presynapse:
            repr_str += f"  LTD_LTP: {self.LTD_LTP}\n"
        repr_str += f"  Postsynapse: {self.postsynapse}\n"
        repr_str += ")"
        return repr_str


class Adaptation(nn.Module):
    def __init__(self, model: nn.Module):
        super(Adaptation, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        #frezzing the model
        for param in self.model.parameters():
                param.requires_grad = False
        print("8888888888888888888888888888888888 Init model Structure 88888888888888888888888888888888888888888")
        total_param_num = sum([m.numel() for m in self.model.parameters()])
        trainable_params = {}
        for name, param in self.model.named_parameters():
            print(f'The name of the module: {name}')
            if 'head' in name or 'norm' in name:
                    param.requires_grad = True
                    trainable_params[name] = param.numel()/total_param_num*100
        print(f'Model parameters count (synapse):  {total_param_num}')
        print(f'Trainable parameters {trainable_params}, total: {sum(trainable_params.values()):.4f} (%)')
        print("8888888888888888888888888888888888 adapt module init 88888888888888888888888888888888888888888")
        self.model.patch_embed.ada = neurotransmitter(768)
        for block_id, block in enumerate(self.model.blocks):
            block.attn.qkv.ada = synapse_plasticity(block.attn.qkv.weight,
                                                    block.attn.qkv.bias,
                                                    self.model.patch_embed.ada
                                                    )
            block.attn.proj.ada = synapse_plasticity(block.attn.proj.weight,
                                                    block.attn.proj.bias,
                                                    ) # type: ignore
            block.mlp.fc1.ada = synapse_plasticity(block.mlp.fc1.weight,
                                                    block.mlp.fc1.bias,
                                                    block.attn.proj.ada.postsynapse
                                                    )
            block.mlp.fc2.ada = synapse_plasticity(block.mlp.fc2.weight,
                                                    block.mlp.fc2.bias,
                                                    block.mlp.fc1.ada.postsynapse
                                                    )
        self.model.head.ada = synapse_plasticity(self.model.head.weight,
                                                self.model.head.bias,
                                                self.model.blocks[-1].mlp.fc2.ada.postsynapse
                                                )
        print("8888888888888888888888888888888888 init finish 88888888888888888888888888888888888888888")
        print(self.model)
        trainable_param_num = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_param_num += param.numel()
        print(f'Trainable parameters total: {trainable_param_num/total_param_num*100:.4f} (%)')
    
    def forward_head_only(self, x):
        return self.model(x)
    
    def forward(self, x, nt_sp_both = 'nt'):
        B, C, H, W = x.shape
        _assert(H == self.model.patch_embed.img_size[0], f"Input image height ({H}) doesn't match model ({self.model.patch_embed.img_size[0]}).")
        _assert(W == self.model.patch_embed.img_size[1], f"Input image width ({W}) doesn't match model ({self.model.patch_embed.img_size[1]}).")
        x = self.model.patch_embed.proj(x) 
        if self.model.patch_embed.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.model.patch_embed.norm(x)
        x = self.model.patch_embed.ada(x)
        if self.model.cls_token is not None:
            x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
    
        for block_id, block in enumerate(self.model.blocks):
            res = x
            x = block.norm1(x)
            B, N, C = x.shape
            qkv = block.attn.qkv.ada(x, nt_sp_both)
            qkv = qkv.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attn = block.attn.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = block.attn.proj.ada(x, nt_sp_both)
            x = block.attn.proj_drop(x)
            x = block.drop_path1(block.ls1(x)) + res
            
            res = x
            x = block.norm2(x)
            x = block.mlp.fc1.ada(x, nt_sp_both)
            x = block.mlp.act(x)
            x = block.mlp.drop1(x)
            x = block.mlp.fc2.ada(x, nt_sp_both)
            x = block.mlp.drop2(x)
            x = block.drop_path2(block.ls2(x)) + res
        x = self.model.norm(x)
        if self.model.global_pool:
            x = x[:, self.model.num_tokens:].mean(dim=1) if self.model.global_pool == 'avg' else x[:, 0]
      
        x = self.model.head.ada(x, nt_sp_both)
        return x
            


def main(args, args_text):
    run_name = f"{args.data_dir.split('/')[-1]}_{args.dataset[0]}_{args.lr}_{args.warmup_lr}_{args.model}_{args.tuning_mode}"
    out_dir = os.path.join(args.output, run_name)
    # skip if output dir exists
    if os.path.exists(out_dir):
        print(f"Output directory {out_dir} already exists, skipping.")
        return
    model, loader_train, loader_eval, mixup_active, mixup_fn = init_model_and_dataloader(args_input)
    args.mixup_active, args.mixup_fn = mixup_active, mixup_fn
    model = Adaptation(model).cuda()
    print("8888888888888888888888888888888888 Optimizer info 88888888888888888888888888888888888888888")
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    print(f'Optimizer: {args.opt}, lr: {args.lr}, weight_decay: {args.weight_decay}')
    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    print(f'LR scheduler: {args.sched}, num_epochs: {num_epochs}')
    print(f'Warmup_epochs: {args.warmup_epochs}, warmup_lr: {args.warmup_lr}, min_lr: {args.min_lr}')
    print("8888888888888888888888888888888888 Loss function 88888888888888888888888888888888888888888")
    train_loss_fn, name = create_loss_fn(args)
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    print('train_loss_fn: ', name)
    print('validate_loss_fn: CrossEntropy loss')
    if args.wandb:
        print("8888888888888888888888888888888888 wandb 88888888888888888888888888888888888888888")
        wandb.login(key=args.wandb)
        wandb.init(project='Neuron and Synapse',
                   config={**vars(args)},
                   name= run_name,
                   reinit=True,
                   )
        wandb.watch(model,
            criterion=train_loss_fn,
            log='all',
            log_freq=15,
            log_graph=True,
            )
    print(f"8888888888888888888888888888888888 Start Training 88888888888888888888888888888888888888888")
    best_result = {'Val_loss': 100, 'Val_acc': 0, 'epoch': 0, 'lr': 0, 'state_dict': model.state_dict()}
    for epoch in range(args.epochs+1):
        if epoch % 20 == 0:
            val_loss, val_acc, val_loss_head_only, val_acc_head_only = validate(model, loader_eval, validate_loss_fn, args)
            print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}%")
            print(f"Validation loss (head only): {val_loss_head_only:.4f}, Validation accuracy (head only): {val_acc_head_only:.4f}%")
            if best_result['Val_acc'] < val_acc:
                best_result['Val_loss'] = val_loss
                best_result['Val_acc'] = val_acc
                best_result['epoch'] = epoch
                best_result['lr'] = optimizer.param_groups[0]['lr']
                out_path = os.path.join(args.output,
                        run_name,
                        f'epoch{best_result["epoch"]}_val{best_result["Val_acc"]:.4f}.pth')
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                torch.save(best_result['state_dict'], out_path)
                print(f"Best model saved at {out_path}")
                
            for name, module in model.named_modules():
                if hasattr(module, 'visualize'):
                    v_out_path = os.path.join(
                        args.output,
                        run_name,
                        f'epoch{epoch}_val{val_acc:.4f}',
                        f'{name}_{module.__class__.__name__}.png'
                    )
                    if not os.path.exists(os.path.dirname(v_out_path)):
                        os.makedirs(os.path.dirname(v_out_path))
                    module.visualize(v_out_path)
            print(f'Visualize result saved at {v_out_path}')
            
            if args.wandb:
                wandb.log({'Val_loss': val_loss, 'Val_acc': val_acc,
                           'Val_loss_head_only': val_loss_head_only, 'Val_acc_head_only': val_acc_head_only,
                           'epoch': epoch,
                           'Best result': best_result})
                # if epoch == 0 or epoch == args.epochs:
                #     artifact = wandb.Artifact(run_name, type='model')
                #     artifact.add_file(out_path)
                #     wandb.log_artifact(artifact)
                
        print(f"Best result: 'Val_loss': {best_result['Val_loss']:.4f}, 'Val_acc': {best_result['Val_acc']:.4f}%, 'epoch': {best_result['epoch']}, 'lr': {best_result['lr']}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, current_lr = train(model, loader_train, optimizer, train_loss_fn, lr_scheduler, args)
        lr_scheduler.step(epoch=epoch)
        if args.wandb:
            wandb.log({'Train_loss': train_loss, 'lr': current_lr, 'epoch': epoch})
            
# VTAB-1k_DATASET = [
#     "cifar_100", "caltech_102", "dtd_47", "oxford_flowers_102", "oxford_iiit_pets_37", "svhn_10", "sun_397",  # Natural
#     "dmlab_6", "patch_camelyon_2", "eurosat_10", "resisc_45", "diabetic_retinopathy_5",  # Specialized
#     "clevr_count_8", "clevr_dist_8", "dmlab_6", "kitti_2", "dsprites_loc_16", "dsprites_ori_16", "smallnorb_azi_18", "smallnorb_ele_18"  # Structured]
# FGVC_DATASET = ["stanforddogs_120", "nabirds_555", "cub2011_200", "oxfordflowers_102", "stanfordcars_196"]
args_input = [
    '--data-dir', '/home/cqzeng/silen/dataset/VTAB', #'/home/cqzeng/silen/dataset/FGVC', '/home/cqzeng/silen/dataset/vtab-1k'
    '--dataset', 'smallnorb_azi_18',
    '--num-classes', '18',
    '--model', 'vit_base_patch16_224_in21k',
    # '--model', 'convnext_base_in22k',
    # '--model', 'as_base_patch4_window7_224',
    # '--model', 'swin_base_patch4_window7_224_in22k',
    '--epochs', '100',
    '--opt', 'adamw',
    '--weight-decay', '0.0',
    '--lr', '0.001',
    '--warmup-lr', '0.0005',
    '--warmup-epochs', '10',
    '--min-lr', '1e-8',
    '--gpu_id', '2',
    '--batch-size', '64',
    '--tuning-mode', 'nt_only',
    '--output', '/home/cqzeng/SAN/output',
    '--wandb', '97e85839e66b93ae618156c2b468f818d4348745',
]

if __name__ == '__main__':
    FGVC_DATASET = ["stanforddogs_120",
                    "nabirds_555",
                    "cub2011_200",
                    "oxfordflowers_102",
                    "stanfordcars_196"
                    ]
    VTAB1k_DATASET = [
        "clevr_count_8",
        "clevr_dist_8",
        "dmlab_6",
        # "kitti_2",
        "dsprites_loc_16",
        "dsprites_ori_16",
        "smallnorb_azi_18",
        "smallnorb_ele_18"  # Structured
        "cifar_100",
        "caltech_102",
        "dtd_47",
        "oxford_flowers_102",
        "oxford_iiit_pets_37",
        "svhn_10",
        "sun_397",  # Natural
        "dmlab_6",
        "dsprites_ori_16",
        "patch_camelyon_2",
        "eurosat_10",
        "resisc_45",
        "diabetic_retinopathy_5",  # Specialized
        ]
    for dataset in VTAB1k_DATASET:
        for lr in [0.005, 0.0025, 0.001, 0.0005]:
            for warmup_lr_ratio in [0.5, 0.25, 0.1]:
                    args_input[3] = dataset
                    args_input[5] = dataset.split('_')[-1]
                    args_input[15] = str(lr)
                    args_input[17] = str(warmup_lr_ratio*lr)
                    args, args_text = _parse_args(args_input)
                    torch.cuda.set_device(f'cuda:{args.gpu_id}')
                    main(args, args_text)