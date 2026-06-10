from __future__ import annotations
from typing import Tuple, Dict, Optional, List


import argparse
import datetime
import os

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from lightning.pytorch.callbacks import (LearningRateMonitor,
                                         ModelCheckpoint, TQDMProgressBar)
from timm.models.layers import DropPath, trunc_normal_
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from data_factory import data_provider
from utils import save_copy_of_files, random_masking_3D, str2bool

from plugins.timae.plugin         import TiMAEPlugin
from plugins.timae.linear_decoder import TiMAELinearPlugin
from plugins.contrast.plugin      import ContrastivePlugin
from plugins.fic.fic              import FisherInformationConstraint


# TSLANet building blocks  (UNCHANGED)
class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop  = nn.Dropout(drop)
        self.act   = nn.GELU()

    def forward(self, x):
        x  = x.transpose(1, 2)
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x  = self.conv3(self.drop(x1) * x2 + self.drop(x2) * x1)
        return x.transpose(1, 2)


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B = x_fft.shape[0]
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        median = energy.view(B, -1).median(dim=1, keepdim=True)[0].view(B, 1)
        norm   = energy / (median + 1e-6)
        mask   = ((norm > self.threshold_param).float()
                  - self.threshold_param).detach() + self.threshold_param
        return mask.unsqueeze(-1)

    def forward(self, x_in):
        B, N, C = x_in.shape
        x  = x_in.to(torch.float32)
        xf = torch.fft.rfft(x, dim=1, norm='ortho')
        w  = torch.view_as_complex(self.complex_weight)
        xw = xf * w
        if args.adaptive_filter:
            fm = self.create_adaptive_high_freq_mask(xf)
            wh = torch.view_as_complex(self.complex_weight_high)
            xw = xw + xf * fm.to(x.device) * wh
        x = torch.fft.irfft(xw, n=N, dim=1, norm='ortho')
        return x.to(x_in.dtype).view(B, N, C)


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.asb       = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.icb       = ICB(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        if args.ICB and args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x

# TSLANet backbone
class TSLANet(nn.Module):

    def __init__(self):
        super().__init__()
        self.patch_size  = args.patch_size
        self.stride      = self.patch_size // 2
        num_patches      = int((args.seq_len - self.patch_size) / self.stride + 1)
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)
        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)
        ])
        self.out_layer = nn.Linear(args.emb_dim * num_patches, args.pred_len)

    def pretrain(self, x_in):
        """Original masking-MSE baseline."""
        x       = rearrange(x_in, 'b l m -> b m l')
        x_patch = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_patch = rearrange(x_patch, 'b m n p -> (b m) n p')
        xb_mask, _, self.mask, _ = random_masking_3D(x_patch,
                                                      mask_ratio=args.mask_ratio)
        self.mask = self.mask.bool()
        xb_mask   = self.input_layer(xb_mask)
        for blk in self.tsla_blocks:
            xb_mask = blk(xb_mask)
        return xb_mask, self.input_layer(x_patch)

    def forward(self, x):
        B, L, M = x.shape
        means  = x.mean(1, keepdim=True).detach()
        x      = x - means
        stdev  = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x     /= stdev
        x      = rearrange(x, 'b l m -> b m l')
        x      = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x      = rearrange(x, 'b m n p -> (b m) n p')
        x      = self.input_layer(x)
        for blk in self.tsla_blocks:
            x = blk(x)
        out = self.out_layer(x.reshape(B * M, -1))
        out = rearrange(out, '(b m) l -> b l m', b=B)
        return out * stdev + means

    def encode(self, x_in: torch.Tensor) -> torch.Tensor:
        B, L, M = x_in.shape
        means  = x_in.mean(1, keepdim=True).detach()
        x      = (x_in - means) / torch.sqrt(
            torch.var(x_in - means, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')   # (B*M, num_patches, patch_size)
        x = self.input_layer(x)                      # (B*M, num_patches, emb_dim)
        for blk in self.tsla_blocks:
            x = blk(x)
        x = x.mean(dim=1)                            # (B*M, emb_dim) 패치 평균
        x = x.reshape(B, M, -1)

        max_channels = min(M, 8) 
        sampled_indices = torch.randperm(M, device=x.device)[:max_channels]
        x_sampled = x[:, sampled_indices, :]         # [B, max_channels, emb_dim]
    
        return x_sampled.reshape(B * max_channels, -1) # 최종 InfoNCE 입력: [B * max_channels, emb_dim]
    


# Pretraining Lightning module
class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model             = TSLANet()
        self.timae_plugin      = None
        self.timae_lin_plugin  = None
        self.contrast_plugin   = None
        self.fic               = None

        if args.pretrain_mode == 'timae':
            self.timae_plugin = TiMAEPlugin(
                encoder=self.model,
                patch_size=args.patch_size,
                stride=args.patch_size // 2,
                seq_len=args.seq_len,
                emb_dim=args.emb_dim,
                mask_ratio=args.timae_mask_ratio,
                decoder_dim=args.timae_decoder_dim,
                decoder_depth=args.timae_decoder_depth,
                decoder_heads=args.timae_decoder_heads,
                norm_pix_loss=args.timae_norm_pix_loss,
            )
            print(f"[Ti-MAE Transformer] mask={args.timae_mask_ratio}")

        elif args.pretrain_mode == 'timae_linear':
            self.timae_lin_plugin = TiMAELinearPlugin(
                encoder=self.model,
                patch_size=args.patch_size,
                stride=args.patch_size // 2,
                seq_len=args.seq_len,
                emb_dim=args.emb_dim,
                mask_ratio=args.timae_mask_ratio,
                norm_pix_loss=args.timae_norm_pix_loss,
            )
            print(f"[Strategy A: Ti-MAE Linear] mask={args.timae_mask_ratio}")

        elif args.pretrain_mode == 'contrastive':
            aug_list = [a.strip() for a in args.augmentations.split(",")
                        if a.strip()]
            self.contrast_plugin = ContrastivePlugin(
                encoder=self.model,
                emb_dim=args.emb_dim,
                proj_dim=args.projection_dim,
                temperature=args.contrastive_temperature,
                init_temperature=args.contrastive_init_temperature,
                warmup_steps=args.contrastive_warmup_steps,
                context_weight=args.context_weight,
                temporal_weight=args.temporal_weight,
                aug_weight=args.aug_weight,
                aug_list=aug_list,
                crop_ratio=args.crop_ratio,
                shift_ratio=args.shift_ratio,
            )
            print(f"[Strategy B: Contrastive] tau={args.contrastive_temperature}  "
                  f"init_tau={args.contrastive_init_temperature}  "
                  f"warmup={args.contrastive_warmup_steps}")

        if args.use_fic:
            fic_params = ['input_layer', 'out_layer'] \
                if getattr(args, 'fic_layerwise', False) else None
            self.fic = FisherInformationConstraint(
                model=self.model,
                lambda_fic=args.fic_lambda,
                param_names=fic_params,
            )
            scope = "input+out layers only" if fic_params else "all params"
            print(f"[Strategy C: FIC] lambda={args.fic_lambda}  scope={scope}")


    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params = list(self.model.parameters())
        if self.timae_plugin is not None:
            params += list(self.timae_plugin.decoder.parameters())
        if self.timae_lin_plugin is not None:
            params += list(self.timae_lin_plugin.decoder.parameters())
        if self.contrast_plugin is not None:
            params += list(self.contrast_plugin.proj_head.parameters())
            # aggregator alpha/beta/gamma 가 nn.Parameter 이므로 등록 필요
            params += list(self.contrast_plugin.aggregator.parameters())
        return optim.AdamW(params, lr=1e-4, weight_decay=1e-6)

    def _calculate_loss(self, batch, mode="train"):
        batch_x = batch[0].float().to(device)

        if self.timae_plugin is not None:
            base_loss, ld = self.timae_plugin(batch_x)
            for k, v in ld.items():
                self.log(f"{mode}_{k}", v,
                         on_step=False, on_epoch=True, logger=True)

        elif self.timae_lin_plugin is not None:
            base_loss, ld = self.timae_lin_plugin(batch_x)
            self.log(f"{mode}_loss_recon", ld["loss_recon"],
                     on_step=False, on_epoch=True, logger=True)

        elif self.contrast_plugin is not None:
            is_training = (mode == "train")
            base_loss, ld = self.contrast_plugin(batch_x, training=is_training)
            for k, v in ld.items():
                self.log(f"{mode}_loss_{k}", v,
                         on_step=False, on_epoch=True, logger=True)

        else:
            preds, target = self.model.pretrain(batch_x)
            lm        = (preds - target) ** 2
            base_loss = (lm.mean(dim=-1) * self.model.mask).sum() \
                      / self.model.mask.sum()

        if self.fic is not None:
            # forward()는 로깅 전용 — 실제 rescaling은 on_after_backward()에서 수행
            total_loss, fi = self.fic(base_loss)
            self.log(f"{mode}_fic_norm", fi["fic_term"],
                     on_step=False, on_epoch=True, logger=True)
            self.log(f"{mode}_task_loss", fi["task_loss"],
                     on_step=False, on_epoch=True, logger=True)
            self.log(f"{mode}_fic_constrained", fi["constrained"],
                     on_step=False, on_epoch=True, logger=True)
        else:
            total_loss = base_loss

        self.log(f"{mode}_loss", total_loss,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def on_after_backward(self):
        if self.fic is not None:
            self.fic.apply_after_backward()

    def training_step(self, batch):
        return self._calculate_loss(batch, "train")

    def validation_step(self, batch):
        self._calculate_loss(batch, "val")

    def test_step(self, batch):
        self._calculate_loss(batch, "test")


# Fine-tuning Lightning module
class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model     = TSLANet()
        self.criterion = nn.MSELoss()
        self.mse       = MeanSquaredError()
        self.mae       = MeanAbsoluteError()
        self.preds     = []
        self.trues     = []
        fic_params = ['input_layer', 'out_layer'] \
            if getattr(args, 'fic_layerwise', False) else None
        self.fic = (FisherInformationConstraint(
                        model=self.model,
                        lambda_fic=args.fic_lambda,
                        param_names=fic_params)
                    if (args.use_fic and args.fic_finetune) else None)
        if self.fic:
            print(f"[FIC fine-tune] lambda={args.fic_lambda}")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        sch = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                   opt, mode='min', factor=0.1, patience=2),
               'monitor': 'val_mse', 'interval': 'epoch', 'frequency': 1}
        return {'optimizer': opt, 'lr_scheduler': sch}

    def _calculate_loss(self, batch, mode="train"):
        bx, by, _, _ = batch
        bx = bx.float().to(device)
        by = by.float().to(device)
        out  = self.model(bx)[:, -args.pred_len:, :]
        by   = by[:, -args.pred_len:, :]
        base = self.criterion(out, by)
        # FIC rescaling은 on_after_backward()에서 수행 — 여기선 loss 변경 없음
        total = base
        pred = out.detach().cpu().contiguous()
        true = by.detach().cpu().contiguous()
        self.log(f"{mode}_loss", total,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mse", self.mse(pred, true),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mae", self.mae(pred, true),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return total, pred, true

    def on_after_backward(self):
        if self.fic is not None:
            self.fic.apply_after_backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

    def training_step(self, batch):
        loss, _, _ = self._calculate_loss(batch, "train")
        return loss

    def validation_step(self, batch):
        self._calculate_loss(batch, "val")

    def test_step(self, batch):
        loss, p, t = self._calculate_loss(batch, "test")
        self.preds.append(p)
        self.trues.append(t)

    def on_test_epoch_end(self):
        p = torch.cat(self.preds).reshape(
            -1, self.preds[0].shape[-2], self.preds[0].shape[-1])
        t = torch.cat(self.trues).reshape(
            -1, self.trues[0].shape[-2], self.trues[0].shape[-1])
        print(f"MAE={self.mae(p,t):.4f}  "
              f"MSE={self.mse(p.contiguous(),t.contiguous()):.4f}")


# Trainer helpers
def pretrain_model():
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH, accelerator="auto",
        devices=1, num_sanity_val_steps=0, max_epochs=args.pretrain_epochs,
        callbacks=[pretrain_ckpt_cb,
                   LearningRateMonitor("epoch"),
                   TQDMProgressBar(refresh_rate=500)],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None
    L.seed_everything(args.seed)
    m = model_pretraining()
    trainer.fit(m, train_loader, val_loader)
    return m, pretrain_ckpt_cb.best_model_path


def train_model(pretrained_path):
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH, accelerator="auto",
        devices=1, num_sanity_val_steps=0, max_epochs=args.train_epochs,
        callbacks=[finetune_ckpt_cb,
                   LearningRateMonitor("epoch"),
                   TQDMProgressBar(refresh_rate=500)],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None
    L.seed_everything(args.seed)

    m = model_training()
    if args.load_from_pretrained and pretrained_path:
        ckpt  = torch.load(pretrained_path, map_location='cpu')
        enc_w = {k: v for k, v in ckpt["state_dict"].items()
                 if k.startswith("model.")}
        miss, unexp = m.load_state_dict(enc_w, strict=False)
        print(f"Loaded {len(enc_w)} encoder keys | "
              f"missing={len(miss)}  unexpected={len(unexp)}")
        for name, p in m.named_parameters():
            if "input_layer" in name:
                print(f"  input_layer mean: {p.mean().item():.6f}")
                break
    elif args.load_from_pretrained and not pretrained_path:
        print("[WARNING] pretrained path empty — training from scratch.")

    trainer.fit(m, train_loader, val_loader)
    best = model_training.load_from_checkpoint(finetune_ckpt_cb.best_model_path)
    val_r  = trainer.test(best, dataloaders=val_loader,  verbose=False)
    test_r = trainer.test(best, dataloaders=test_loader, verbose=False)
    return best, \
           {"test": test_r[0]["test_mse"], "val": val_r[0]["test_mse"]}, \
           {"test": test_r[0]["test_mae"], "val": val_r[0]["test_mae"]}


# CLI
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data',      type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='data/ETT-small')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--embed',     type=str, default='timeF')
    parser.add_argument('--features',  type=str, default='M')
    parser.add_argument('--target',    type=str, default='OT')
    parser.add_argument('--freq',      type=str, default='h')
    parser.add_argument('--seq_len',   type=int, default=512)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len',  type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

    # Training
    parser.add_argument('--train_epochs',    type=int,   default=20)
    parser.add_argument('--pretrain_epochs', type=int,   default=20)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--seed',            type=int,   default=42)

    # Model
    parser.add_argument('--emb_dim',    type=int,   default=64)
    parser.add_argument('--depth',      type=int,   default=2)
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--patch_size', type=int,   default=16)
    parser.add_argument('--mask_ratio', type=float, default=0.4)

    # TSLANet flags
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True)
    parser.add_argument('--ICB',             type=str2bool, default=True)
    parser.add_argument('--ASB',             type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # Pretrain mode
    parser.add_argument('--pretrain_mode', type=str, default='masking',
        choices=['masking', 'timae', 'timae_linear', 'contrastive'])

    # Ti-MAE shared
    parser.add_argument('--timae_mask_ratio',    type=float,    default=0.75)
    parser.add_argument('--timae_norm_pix_loss', type=str2bool, default=True)

    # Ti-MAE Transformer decoder
    parser.add_argument('--timae_decoder_dim',   type=int, default=64)
    parser.add_argument('--timae_decoder_depth', type=int, default=2)
    parser.add_argument('--timae_decoder_heads', type=int, default=4)

    # Strategy C: FIC
    parser.add_argument('--use_fic',      type=str2bool, default=False)
    parser.add_argument('--fic_lambda',   type=float,    default=0.1)
    parser.add_argument('--fic_finetune', type=str2bool, default=False)
    parser.add_argument('--fic_layerwise', type=str2bool, default=False,
                        help='FIC를 input_layer+out_layer에만 적용. '
                             'Default False = 전체 파라미터 (FIC 논문과 일치).')

    # Strategy B: Contrastive
    parser.add_argument('--context_weight',          type=float, default=1.0)
    parser.add_argument('--temporal_weight',         type=float, default=1.0)
    parser.add_argument('--aug_weight',              type=float, default=1.0)
    parser.add_argument('--contrastive_temperature', type=float, default=0.2)
    parser.add_argument('--contrastive_init_temperature', type=float, default=0.5,
                        help='Warmup 시작 temperature. 높을수록 초기 gradient 가 안정적.')
    parser.add_argument('--contrastive_warmup_steps',     type=int,   default=500,
                        help='Temperature warmup step 수. '
                             '데이터셋이 작으면 낮게(200), 크면 높게(1000) 조정.')
    parser.add_argument('--projection_dim',          type=int,   default=128)
    parser.add_argument('--augmentations', type=str, default='jitter,scaling')
    parser.add_argument('--crop_ratio',  type=float, default=0.75)
    parser.add_argument('--shift_ratio', type=float, default=0.10)

    args   = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fic_tag  = f"_FIC{args.fic_lambda}" if args.use_fic else ""
    run_desc = (
        f"{args.data}_{args.data_path.split('.')[0]}"
        f"_emb{args.emb_dim}_d{args.depth}_ps{args.patch_size}"
        f"_pl{args.pred_len}_bs{args.batch_size}"
        f"_mode_{args.pretrain_mode}{fic_tag}"
        f"_{datetime.datetime.now().strftime('%H_%M')}"
    )
    print(f"\n{'='*60}\n  {run_desc}\n{'='*60}")

    CHECKPOINT_PATH = f"lightning_logs/{run_desc}"
    pretrain_ckpt_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH, save_top_k=1,
        filename='pretrain-{epoch}', monitor='val_loss', mode='min')
    finetune_ckpt_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH, save_top_k=1, monitor='val_mse', mode='min')
    save_copy_of_files(finetune_ckpt_cb)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    _, train_loader = data_provider(args, flag='train')
    _, val_loader   = data_provider(args, flag='val')
    _, test_loader  = data_provider(args, flag='test')
    print("Dataset loaded ...")

    # 1. 사용자가 터미널(CLI)로 입력한 원래 FIC 설정을 백업합니다.
    original_use_fic = args.use_fic
    original_fic_lambda = args.fic_lambda

    # 2. 사전 학습(Pre-training) 단계: 강제로 FIC 끄기
    if args.load_from_pretrained:
        if original_use_fic:
            print("\n[Auto-Config] 사전 학습 중에는 FIC를 강제로 비활성화합니다 (일반화 뼈대 구축 및 충돌 방지).")
        args.use_fic = False
        args.fic_lambda = 0.0
        _, best_model_path = pretrain_model()
    else:
        best_model_path = ''

    # 3. 파인튜닝(Fine-tuning) 단계: FIC 원래 설정으로 켜기
    if original_use_fic:
        print(f"\n[Auto-Config] 파인튜닝 단계에서 FIC를 활성화합니다. (lambda={original_fic_lambda}, layerwise={args.fic_layerwise})")
        args.use_fic = True
        args.fic_lambda = original_fic_lambda
        args.fic_finetune = True  # 파인튜닝 모듈(model_training)에서 FIC가 작동하도록 플래그 ON

    model, mse_result, mae_result = train_model(best_model_path)
    print(f"\nMSE: {mse_result}\nMAE: {mae_result}")

    df = pd.DataFrame({'MSE': mse_result, 'MAE': mae_result})
    df.to_excel(os.path.join(CHECKPOINT_PATH,
        f"results_{datetime.datetime.now().strftime('%H_%M')}.xlsx"))

    os.makedirs("textOutput", exist_ok=True)
    fname = f"textOutput/TSLANet_{args.data}_{args.data_path}.txt"
    with open(fname, 'a') as f:
        f.write(run_desc + "\n")
        f.write(f"MSE:{mse_result}, MAE:{mae_result}\n\n")
