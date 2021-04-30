import faulthandler
faulthandler.enable()
import tensorflow as tf
import argparse
import os
from os.path import *
import torch
#from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
from data import data_helper
## from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler, FocalLoss
from utils.Logger import Logger
import numpy as np
from models.resnet import resnet18, resnet50
from tqdm import tqdm
from utils.utils import AUCMeter

_save_models_dir = join(dirname(__file__), 'save_models')

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #hardware
    parser.add_argument("--cuda_number", choices=[i for i in range(torch.cuda.device_count())], help="Choose use which cuda", default=0, type=int)
    #dataset to use
    parser.add_argument("--source", type=str, default='', help="Source, use '-' to connect")
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #criterion
    parser.add_argument("--loss", default='ce', type=str, help="loss function")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--downsample_target", action='store_true', help="Use downsampled target to evaluate")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--start_epoch", type=int, default=0, metavar='N',
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    #misc
    parser.add_argument("--eval_mode", action='store_true', help="Launch evaluation mode(default:False)")
    parser.add_argument("--save_model", action='store_true', help="Save model's params")
    parser.add_argument("--RSC_flag", action='store_true', help="Whether use RSC, default True")
    parser.add_argument("--pretrained", action='store_true', help="Whether use Pretrained model, default True")
    parser.add_argument("--resume",default='',type=str,metavar='PATH',help="path to latest checkpoint(default:none)")
    parser.add_argument("--infer_model", default='', type=str, metavar='PATH',help="Inference model's path(default:none)")
    return parser.parse_args()
    
class Trainer:
    def __init__(self, args, device): 
        self.args = args
        self.device = device
        if args.network == 'resnet18':
            model = resnet18(pretrained=self.args.pretrained, classes=args.n_classes)
        elif args.network == 'resnet50':
            model = resnet50(pretrained=self.args.pretrained, classes=args.n_classes)
        else:
            model = resnet18(pretrained=self.args.pretrained, classes=args.n_classes)
        self.model = model.to(device)
        
        if args.resume:
            if isfile(args.resume):
                print(f"=> loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)
                self.args.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model'])
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                raise ValueError(f"Failed to find checkpoint {args.resume}")
                
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        # self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_tgt_dataloader(self.args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (
        len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all,nesterov=args.nesterov)
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
        self.topk = [0 for _ in range(3)]

    def _do_epoch(self, epoch=None):
        if self.args.loss == 'ce':
            criterion = nn.CrossEntropyLoss()
        elif self.args.loss == 'fl':
            criterion = FocalLoss(class_num = self.args.n_classes)
        self.model.train()
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            self.optimizer.zero_grad()

            data_flip = torch.flip(data, (3,)).detach().clone()
            data = torch.cat((data, data_flip))
            class_l = torch.clamp(class_l, 0, 9)
            class_l = torch.cat((class_l, class_l))

            class_logit = self.model(data, class_l, self.args.RSC_flag, epoch)
            class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            loss = class_loss

            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {"loss": class_loss.item()},
                            {"class": torch.sum(cls_pred == class_l.data).item(), }, data.shape[0])
            del loss, class_loss, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct,auc_dict = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class_acc": class_acc})
                self.logger.log_test(phase, {"auc": auc_dict['auc']})
                self.logger.log_test(phase, {"fpr_980": auc_dict['fpr_980']})
                self.logger.log_test(phase, {"fpr_991": auc_dict['fpr_991']})
                self.results[phase][self.current_epoch] = class_acc
                    
                #save best&latest model params
                if phase == 'val':
                    self.save_model(epoch, auc_dict)
                del auc_dict


    def do_test(self, loader):
        class_correct = 0
        auc_meter = AUCMeter()
        for it, ((data, nouse, class_l), _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            class_logit = self.model(data, class_l, False)
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

            cls_score = F.softmax(class_logit,dim=1)
            auc_meter.update(class_l.cpu(),cls_score.cpu())

        auc, fpr_980, fpr_991, fpr_993, fpr_995, fpr_997, fpr_999, fpr_1, thresholds = auc_meter.calculate()
        auc_dict = {'auc':auc, 'fpr_980':fpr_980, 'fpr_991':fpr_991, 'fpr_993':fpr_993, 'fpr_995':fpr_995, 'fpr_997':fpr_997, 'fpr_999':fpr_999,'fpr_1':fpr_1,'thresholds':thresholds}
        return class_correct, auc_dict

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=50)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.start_epoch, self.args.epochs):
            self._do_epoch(self.current_epoch)
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_last_lr())
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model

    # def save_model(self,epoch, auc_dict):
    #     if not exists(_save_models_dir): os.mkdir(_save_models_dir)
    #     state_to_save = {'model':self.model.state_dict(), 'auc_dict':auc_dict, 'epoch':epoch}
    #     tmp_auc, tmp_fpr_980 = auc_dict['auc'], auc_dict['fpr_980']
    #     best1,best2,best3 = self.moving_record['best1'],self.moving_record['best2'],self.moving_record['best3']
    #     best1_path, best2_path, best3_path = (join(_save_models_dir, f"tgt_{self.args.target}_src_{'-'.join(self.args.source)}_RSC_{self.args.RSC_flag}_best{_}.pth") for _ in [1,2,3])
    #     #resort top3
    #     update_pos = -1
    #     if tmp_auc>best1['auc']:
    #         best3['auc'], best3['fpr_980'] = best2['auc'], best2['fpr_980']
    #         best2['auc'], best2['fpr_980'] = best1['auc'], best1['fpr_980']
    #         best1['auc'], best1['fpr_980'] = tmp_auc, tmp_fpr_980
    #         if exists(best2_path) and exists(best3_path):
    #             os.rename(best2_path, best3_path)
    #         if exists(best1_path) and exists(best2_path):
    #             os.rename(best1_path, best2_path)
    #         update_pos = 1
    #     elif best2['auc']< tmp_auc < best1['auc']:
    #         best3['auc'], best3['fpr_980'] = best2['auc'], best2['fpr_980']
    #         best2['auc'], best2['fpr_980'] = tmp_auc, tmp_fpr_980
    #         if exists(best2_path) and exists(best3_path):
    #             os.rename(best2_path, best3_path)
    #         update_pos = 2
    #     elif best3['auc']< tmp_auc < best2['auc']:
    #         best3['auc'], best3['fpr_980'] = tmp_auc, tmp_fpr_980
    #         update_pos = 3
        
    #     if update_pos in [1,2,3]:
    #         model_saved_path = join(_save_models_dir, f"tgt_{self.args.target}_src_{'-'.join(self.args.source)}_RSC_{self.args.RSC_flag}_best{update_pos}.pth")
    #         torch.save(state_to_save, model_saved_path)
    #         print(f'=>Best{update_pos} model updated and saved in path {model_saved_path}')
    #     if epoch in range(self.args.epochs-3, self.args.epochs):
    #         model_saved_path = join(_save_models_dir, f"tgt_{self.args.target}_src_{'-'.join(self.args.source)}_RSC_{self.args.RSC_flag}_epochs{epoch}.pth")
    #         torch.save(state_to_save, model_saved_path)
    #         print(f'=>Last{self.args.epochs - epoch} model updated and saved in path {model_saved_path}')
    def save_model(self,epoch,auc_dict):
        if not exists(_save_models_dir): os.mkdir(_save_models_dir)
        tmp_auc, tmp_fpr_980 = auc_dict['auc'], auc_dict['fpr_980']
        for i,rec in enumerate(self.topk):
            if tmp_auc > rec:
                for j in range(len(self.topk)-1,i,-1):
                    self.topk[j] = self.topk[j-1]
                    _j, _jm1 = join(_save_models_dir, f"tgt_{self.args.target}_src_{'-'.join(self.args.source)}_RSC_{self.args.RSC_flag}_best{j+1}.pth"),\
                    join(_save_models_dir, f"tgt_{self.args.target}_src_{'-'.join(self.args.source)}_RSC_{self.args.RSC_flag}_best{j}.pth")
                    if exists(_jm1):
                        os.rename(_jm1,_j)
                self.topk[i] = tmp_auc
                model_saved_path = join(_save_models_dir, f"tgt_{self.args.target}_src_{'-'.join(self.args.source)}_RSC_{self.args.RSC_flag}_best{i+1}.pth")
                state_to_save = {'model':self.model.state_dict(), 'auc_dict':auc_dict, 'epoch':epoch}
                torch.save(state_to_save, model_saved_path)
                print(f'=>Best{i+1} model updated and saved in path {model_saved_path}')
                break

        if epoch in range(self.args.epochs-3, self.args.epochs):
            model_saved_path = join(_save_models_dir, f"tgt_{self.args.target}_src_{'-'.join(self.args.source)}_RSC_{self.args.RSC_flag}_epochs{epoch}.pth")
            torch.save(state_to_save, model_saved_path)
            print(f'=>Last{self.args.epochs - epoch} model updated and saved in path {model_saved_path}')

class Infer:
    def __init__(self,args,device):
        self.args = args
        self.device = device
        if args.network == 'resnet18':
            model = resnet18(pretrained=self.args.pretrained, classes=args.n_classes)
        elif args.network == 'resnet50':
            model = resnet50(pretrained=self.args.pretrained, classes=args.n_classes)
        else:
            model = resnet18(pretrained=self.args.pretrained, classes=args.n_classes)
        self.model = model.to(device)
        self.model_params_PATH = None
    
        if args.infer_model:
            self.model_params_PATH = args.infer_model
            if isfile(self.model_params_PATH):
                print(f"=> loading checkpoint '{self.model_params_PATH}'")
                checkpoint = torch.load(self.model_params_PATH)
                state_dict = checkpoint['model']
                #try to fix last fc layer's name dismatch ['fc'->'class_classifier']
                for key in list(state_dict.keys()):
                    if key == 'fc.weight':
                        state_dict['class_classifier.weight'] = state_dict.pop('fc.weight')
                    elif key == 'fc.bias':
                        state_dict['class_classifier.bias'] = state_dict.pop('fc.bias')
                #load state_dict
                self.model.load_state_dict(state_dict)
                print(f"=> loaded checkpoint")
            else:
                raise ValueError(f"Failed to find checkpoint {self.model_params_PATH}")
        self.dataloader = data_helper.get_tgt_dataloader(self.args, patches=self.model.is_patch_based())
    
    def eval(self):
        self.model.eval()
        self.logger = Logger(self.args)
        with torch.no_grad():
            total = len(self.dataloader.dataset)
            class_correct, auc_dict = self.do_test(self.dataloader)
            class_acc = float(class_correct)/total
            self.logger.log_test('Inference Result', {'class_acc':class_acc})
            self.logger.log_test('Inference Result', {'auc':auc_dict['auc']})
            self.logger.log_test('Inference Result', {'fpr_980':auc_dict['fpr_980']})
            self.logger.log_test('Inference Result', {'fpr_991':auc_dict['fpr_991']})
            del auc_dict

    def do_test(self, loader):
        class_correct = 0
        auc_meter = AUCMeter()
        for it, ((data, nouse, class_l), _) in enumerate(tqdm(loader)):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            class_logit = self.model(data, class_l, False)
            _, cls_pred = class_logit.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

            cls_score = F.softmax(class_logit,dim=1)
            auc_meter.update(class_l.cpu(),cls_score.cpu())

        auc, fpr_980, fpr_991, fpr_993, fpr_995, fpr_997, fpr_999, fpr_1, thresholds = auc_meter.calculate()
        auc_dict = {'auc':auc, 'fpr_980':fpr_980, 'fpr_991':fpr_991, 'fpr_993':fpr_993, 'fpr_995':fpr_995, 'fpr_997':fpr_997, 'fpr_999':fpr_999,'fpr_1':fpr_1,'thresholds':thresholds}
        return class_correct, auc_dict

def main():
    args = get_args()
    print("Environment:")
    print(f"\tNum of Epochs: {args.epochs}")
    print(f"\tBatch Size: {args.batch_size}")
    print(f"\tBackbone: {args.network}")
    print(f"\tUsing RSC: {args.RSC_flag}")
    print(f"\tUsing Pretrained: {args.pretrained}")
    # --------------------------------------------
    print("Source domains: {}".format(args.source))
    print("Target domain: {}".format(args.target))
    args.source = args.source.split('-')
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.cuda_number)
    print('Current use cuda:%d' % args.cuda_number)
    if args.eval_mode:
        infer = Infer(args, device)
        infer.eval()
    else:
        trainer = Trainer(args, device)
        trainer.do_training()
    

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
