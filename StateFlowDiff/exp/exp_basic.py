import os
import torch
from StateFlowDiff.models import StateFlowDiff as HATEK
from StateFlowDiff.utils.print_args import print_model_param_stats


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'HATEK': HATEK,
            'StateFlowDiff': HATEK,
            'RegDiff': HATEK,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        print_model_param_stats(self.model, self.args.model)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cuda:{}'.format(self.args.gpu))
                torch.cuda.set_device(device)
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
