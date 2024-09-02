import torch
import numpy as np


from nnunetv2.utilities.helpers import dummy_context
from torch.amp import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from torch._dynamo import OptimizedModule
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
#from .grokfast import gradfilter_ma, gradfilter_ema


# Didnt work
"""
class nnUNetTrainer_autopetiii(nnUNetTrainer):
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        black_magic = [-1  if 'psma' in i else 1 for i in batch['keys']]
        for i in range(self.batch_size):
            data[i,...] = data[i, ...] * black_magic[i]
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
"""

class Masked_DC_and_CE_loss(DC_and_CE_loss):
    def forward(self, net_output: torch.Tensor, target: torch.Tensor, loss_mask):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        if mask is not None:
            loss_mask = loss_mask | mask
        dc_loss = self.dc(net_output, target_dice, loss_mask=loss_mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class nnUNetTrainer_autopetiii(nnUNetTrainer):

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = Masked_DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def on_train_start(self):
        res = super().on_train_start()
        #self.grads = None
        return res

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(device=self.device, non_blocking=True) for i in target]
            mask = [i>0 for i in target]
        else:
            target = target.to(device=self.device, non_blocking=True)
            mask = target>0

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target, mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            #self.grads = gradfilter_ema(mod, grads=self.grads, alpha=.98, lamb=2.)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            #self.grads = gradfilter_ema(mod, grads=self.grads, alpha=.98, lamb=2.)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}


    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(device=self.device, non_blocking=True) for i in target]
            mask = [i>0 for i in target]
        else:
            target = target.to(device=self.desvice, non_blocking=True)
            mask = target>0

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target, mask)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
