import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset.DataGen import Peta, Rap, PA100K, UPAR
from module import backbone
from module.loss_function import adjLoss, WeightLoss, smooth_BCELoss, WeightMSELoss
from module.testnetwork_merge import Framework_merge_base, Framework_merge_test4
from utils.matrics import calculate_accuracy, calcute2, calcute3, calculate_accuracy_in_detail
from utils.model_ema import ModelEma
from torchvision import transforms as T
import torchsummary
import time

from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_


optim_collection = {'Adam': optim.Adam, 'SGD': optim.SGD}
criterion_collection = {'BCE': nn.BCEWithLogitsLoss, 'MSE': nn.MSELoss, 'Weight': WeightLoss, 'Weight_MSE': WeightMSELoss, 'smooth': smooth_BCELoss}


class Manager(object):

    def __init__(self, args):
        self.save_epoch = -1
        self.detail = args.detail
        self.dataNameForDetail = args.dataset#
        self.epoch = 0
        restore = False
        cuda_list = args.cuda
        lr = args.lr
        train_step = args.train_step
        self.SubNet = args.SubNet
        self.learn_gcn = args.learn_gcn
        self.train_step = args.train_step
        self.mark = args.mark
        if self.dataNameForDetail == 'PETA':
            self.train_data = train_data = Peta(args, train=True)
            val_data = Peta(args, train=False)
        elif self.dataNameForDetail == 'RAP':
            self.train_data = train_data = Rap(args, train=True)
            val_data = Rap(args, train=False)
        elif self.dataNameForDetail == 'PA100k':
            self.train_data = train_data = PA100K(args, train=True)
            val_data = PA100K(args, train=False)
        self.bs= args.batch_size
        self.bs2= args.batch_size2
        self.dataloader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                           drop_last=True)
        self.dataloader_val = DataLoader(val_data, batch_size=args.batch_size2, shuffle=False, num_workers=4,
                                         drop_last=True)
        bb = backbone.backbone_collection[args.backbone]
        test = 0
        self.test = test

        self.head_key_index = [0, 1, 2, 3, 4]
        self.arm_key_index = [7, 8, 9, 10]
        self.upper_key_point = [5, 6, 11, 12]
        self.lower_key_point = [11, 12, 13, 14]
        self.foot_key_point = [15, 16]

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.train_number = 49

        print('***************Framework_test*********************')
        
        #resnet : 2048 7 7 1024 14 14 7
        #resnet : 2048 7 7 256 56 56 5
        #convbase: 1024 7 7 256 28 28 3
        #convbase: 1024 7 7 512 14 14 5
        self.net = Framework_merge_test4(
        #self.net = Framework_merge_base(
            backbone=bb,
            num_classes=train_data.dataset.classes_num,
            in_channel=1024,
            height=7,
            width=7,
            in_channel2=512,
            height2=14,
            width2=14,
            stage=5,
            number=self.train_number,
            group = self.train_data.key_group,
            num_gcn=args.GCN,
            embedding_se=False,
        )
        

        self.ema_m = ModelEma(self.net, args.ema_decay) # 0.9997
        self.optimizer = optim_collection[args.optim](params=self.net.parameters(), lr=args.lr,weight_decay=0.0)
        self.loss_weight = self.get_weight(train_data.dataset.label_train)
        if args.cuda:
            self.net = self.net.cuda()
            self.ema_m = self.ema_m.cuda()
            if len(args.cuda) > 1:
                self.net = torch.nn.DataParallel(self.net)
        if args.saved_path != '':
            save_path = args.saved_path
            self.epoch = 0
            print('restore from %s' % args.saved_path)
            restore_dict = self.load(save_path)
            args = restore_dict['args']
            args.cuda = cuda_list
            args.lr = lr
            args.train_step = train_step
            restore = True
        if restore:
            self.net.load_state_dict(restore_dict['model_state_dict'])
        if args.cuda:
            self.loss_weight = self.loss_weight.cuda()
        self.criterion = criterion_collection[args.criterion](self.loss_weight)
        self.cri_pre = criterion_collection['BCE']()
        self.criterion_ft = criterion_collection['DEBCE'](sample_weight=None, scale=1, size_sum=32)
        if args.cuda:
            self.criterion = self.criterion.cuda()

        if self.learn_gcn:
            self.learn_gcn_criterion = adjLoss(train_data.dataset.classes_num)

        if restore:
            self.net.load_state_dict(restore_dict['model_state_dict'])
            self.optimizer.load_state_dict(restore_dict['optim_state_dict'])
            self.max_mA = restore_dict['acc_dict']['mA']
        else:
            self.max_mA = 0
        self.save_rootpath = './saved_module'
        self.save_module_path = './module/' 
        self.args = args
        path = self.mkdir_save(self.save_rootpath)
        logging.basicConfig(filename=path + '/log.log', filemode='a', level=logging.INFO)

        self.log_mark = False

        self.trans = T.Compose([
            T.Resize((args.resolution, args.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.gap = 5
        self.gap_start = 10
        self.cross = True

    def get_weight(self, label_train):
        weight = np.sum(label_train, axis=0)
        weight = weight / len(label_train)
        weight = torch.tensor(weight, dtype=torch.float32)
        return weight


    def train(self):

        scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.dataloader_train), epochs=self.args.train_step, pct_start=0.2)
        scaler = torch.cuda.amp.GradScaler(enabled=False) 
        best_epoch = -1
        best_regular_mA = 0
        best_regular_epoch = -1
        best_ema_mA = 0
        best_mA = 0
        regular_mA_list = []
        ema_mA_list = []
        torch.cuda.empty_cache()
        for i in range(self.epoch, self.args.train_step):
            epoch = i
                
            if self.args.ema_epoch == epoch:
                self.ema_m = ModelEma(self.net, self.args.ema_decay)
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            
            if self.cross:
                if epoch >= self.gap_start:
                    if (epoch//self.gap) % 2 == 0:
                        for p in self.net.parameters():
                            p.requires_grad=False
                        for p in self.net.features[0:7].parameters():
                            p.requires_grad=True
                    else:
                        for p in self.net.parameters():
                            p.requires_grad=True
                        for p in self.net.features[0:7].parameters():
                            p.requires_grad=False

            self.train_framework_epoch(scheduler, scaler, epoch)
            torch.cuda.empty_cache()
            mA = self.Framework_eval(i)
            mA_ema = self.Framework_eval_ema(i)
            torch.cuda.empty_cache()
            regular_mA_list.append(mA)
            ema_mA_list.append(mA_ema)
            if mA > best_regular_mA:
                best_regular_mA = max(best_regular_mA, mA)
                best_regular_epoch = epoch
            if mA_ema > best_ema_mA:
                best_ema_mA = max(mA_ema, best_ema_mA)
            
            if mA_ema > mA:
                mA = mA_ema
                state_dict = self.ema_m.module.state_dict()
            else:
                state_dict = self.net.state_dict()
            is_best = mA > best_mA
            if is_best:
                best_epoch = epoch
            best_mA = max(mA, best_mA)

            log_str = 'epoch: %d | %s\tin ep %d \t mA: %.4f' % (
                epoch + 1, 'Best mA', best_epoch+1, best_mA)
            print(log_str)
            logging.info(log_str)
            log_str = '\t\t| %s\tin epoch: %d\t mA: %.4f' % (
                'best regular mA', best_regular_epoch + 1, best_regular_mA)
            print(log_str)
            logging.info(log_str)


    def train_framework_epoch(self, scheduler, scaler, epoch):
        self.net.train()
        for j, (img, keypoints, labels) in enumerate(tqdm(self.dataloader_train)):
            img = Variable(img.cuda())
            labels = Variable(labels.cuda())
            keypoints = Variable(keypoints.cuda())
            self.optimizer.zero_grad()

            outs = self.net(img, keypoints)
            outs_new = []

            outs_new.append(outs[4])
            outs = outs_new
    
            for i in range(len(outs) - 1, -1, -1):
                retain_graph = i > 0
                if i == len(outs) - 1 and self.learn_gcn:
                    loss_tmp = self.learn_gcn_criterion(outs[i], labels)
                    scaler.scale(loss_tmp).backward(retain_graph=retain_graph)
                else:
                    if outs[i]!=None:
                        loss_tmp = self.criterion(outs[i], labels)

                        scaler.scale(loss_tmp).backward(retain_graph=retain_graph)

            
            clip_grad_norm_(self.net.parameters(), max_norm=10.0)


            scaler.step(self.optimizer)
            scaler.update()
            scheduler.step()
            if epoch >= self.args.ema_epoch:
                self.ema_m.update(self.net)

    
    def Framework_eval(self, epoch):

        self.net.eval()
        for i, (img, keypoints, labels) in enumerate(tqdm(self.dataloader_val)):
            img = Variable(img.cuda())
            labels = Variable(labels.cuda())
            keypoints = Variable(keypoints.cuda())
            outs = self.net(img, keypoints)
            labels = outs[-1]
            outs = outs[:-1]
            outs_new = []

            outs_new.append(outs[4])
            outs = outs_new
            emb_pre = (torch.sigmoid(outs[-1]) > 0.5).cpu().detach().numpy()
            if i == 0:
                emb_pre_arr = emb_pre.copy()
            else:
                emb_pre_arr = np.vstack((emb_pre_arr, emb_pre))


            labels = labels.cpu().numpy()
            if i == 0:
                val_label_arr = labels.copy()
            else:
                val_label_arr = np.vstack((val_label_arr, labels))

        print(
            '***********************************************************************************************************')
        mA = self.log(epoch, val_label_arr, emb_pre_arr, 'emb')
        return mA

    def Framework_eval_ema(self, epoch):

        self.ema_m.eval()
        for i, (img, keypoints, labels) in enumerate(tqdm(self.dataloader_val)):
            img = Variable(img.cuda())
            labels = Variable(labels.cuda())
            keypoints = Variable(keypoints.cuda())
            outs = self.ema_m.module(img, labels, keypoints)
            labels = outs[-1]
            outs = outs[:-1]
            outs_new = []

            outs_new.append(outs[4])
            outs = outs_new
            emb_pre = (torch.sigmoid(outs[-1]) > 0.5).cpu().detach().numpy()
            if i == 0:
                emb_pre_arr = emb_pre.copy()
            else:
                emb_pre_arr = np.vstack((emb_pre_arr, emb_pre))
            
            labels = labels.cpu().numpy()
            if i == 0:
                val_label_arr = labels.copy()
            else:
                val_label_arr = np.vstack((val_label_arr, labels))

        print(
            '***********************************************************************************************************')
        
        mA_ema = self.log(epoch, val_label_arr, emb_pre_arr, 'emb_ena')
        return mA_ema
        

    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self, name, fmt=':f', val_only=False):
            self.name = name
            self.fmt = fmt
            self.val_only = val_only
            self.reset()
    
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
    
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
        def __str__(self):
            if self.val_only:
                fmtstr = '{name} {val' + self.fmt + '}'
            else:
                fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)
    
    
    class AverageMeterHMS(AverageMeter):
        """Meter for timer in HH:MM:SS format"""
        def __str__(self):
            if self.val_only:
                fmtstr = '{name} {val}'
            else:
                fmtstr = '{name} {val} ({sum})'
            return fmtstr.format(name=self.name, 
                                 val=str(datetime.timedelta(seconds=int(self.val))), 
                                 sum=str(datetime.timedelta(seconds=int(self.sum))))
    
    class ProgressMeter(object):
        def __init__(self, num_batches, meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix
    
        def display(self, batch, logger):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            logger.info('  '.join(entries))
    
        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return '[' + fmt + '/' + fmt.format(num_batches) + ']'
        
                
    def log(self, epoch, val_label_arr, val_pre_arr, info=''):
        from utils.matrics import mA_F
        acc, prec, rec, F1, mA, mA_detail = calculate_accuracy_in_detail(val_label_arr, val_pre_arr)#
        mA2 = mA_F(val_label_arr, val_pre_arr)
        log_str = '%s\t%s epoch: %d\t mA: %.4f \t mA2: %.4f' % (
            info, str(datetime.now().strftime('%H:%M:%S')), epoch + 1, mA, mA2)
        print(log_str)
        logging.info(log_str)
                
        acc_dict = {'mA': mA}
        subnet = 'SubNet' if self.args.SubNet else '_'
        save_name = self.args.backbone + '_' + info + '_' + str(epoch) + subnet + '_' + str(mA)
        path = self.mkdir_save(self.save_module_path)
        ori_path = path
        path_pre = os.path.join(ori_path, save_name + '_pre.npy')
        np.save(path_pre, val_pre_arr)
        if self.save_epoch != epoch:
            self.save(acc_dict, save_name)
            self.save_ema(acc_dict, save_name)
            self.save_epoch = epoch
        return mA



    def mkdir_save(self, rootpath):
        if not os.path.exists(rootpath):
            os.mkdir(rootpath)
        path = os.path.join(rootpath, datetime.now().strftime("%Y_%m_%d"))

        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, self.args.information)
        if not os.path.exists(path):
            os.mkdir(path)

        return path

    def save(self, acc_dict, save_name):

        path = self.mkdir_save(self.save_module_path)
        if not self.log_mark:
            logging.info(path)
            self.log_mark = True
        path = os.path.join(path, save_name + '.pt')

        self.net.cpu()
        save_dict = {
            'args': self.args,
            'acc_dict': acc_dict,
            'model_state_dict': self.net.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        }
        if self.args.cuda:
            self.net.cuda()
        torch.save(save_dict, path)
    
    def save_ema(self, acc_dict, save_name):

        path = self.mkdir_save(self.save_module_path)
        if not self.log_mark:
            logging.info(path)
            self.log_mark = True
        path = os.path.join(path, save_name + '_ema.pt')

        self.ema_m.module.cpu()
        save_dict = {
            'args': self.args,
            'acc_dict': acc_dict,
            'model_state_dict': self.ema_m.module.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        }
        if self.args.cuda:
            self.ema_m.module.cuda()
        torch.save(save_dict, path)

    def load(self, save_path):
        save_dict = torch.load(save_path)
        return save_dict

    
    def Framework_predict(self, img, keypoints, check = False):
        self.net.eval()
        img = Variable(img.to(self.device)).unsqueeze(dim=0)
        keypoints = keypoints.unsqueeze(dim=0).detach()
        labels = torch.zeros((img.shape[0], 44))
        labels = labels.cuda()
        if check:
            torchsummary.summary(self.net,input_size=[(3,224,224), (17,3)],)

            memory_summary = torch.cuda.memory_summary(device=None, abbreviated=False)
            print("MODEL_____memoty:", memory_summary)

            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            
            start_event.record()
            outs = self.net(img, keypoints)
            end_event.record()
            torch.cuda.synchronize()

            forward_time = start_event.elapsed_time(end_event)
            print(f"MODEL_____Forward time: {forward_time} milliseconds")
        else:
            outs = self.net(img, keypoints)
        out = torch.sigmoid(outs[0])
        out = out.detach().cpu().numpy().squeeze()

        return out
    
    def Framework_predict_feature(self, img, keypoints, check = False):
        self.net.eval()
        img = Variable(img.to(self.device)).unsqueeze(dim=0)
        keypoints = keypoints.unsqueeze(dim=0).detach()
        labels = torch.zeros((img.shape[0], 44))
        labels = labels.cuda()
        if check:
            torchsummary.summary(self.net,input_size=[(3,224,224), (17,3)],)

            memory_summary = torch.cuda.memory_summary(device=None, abbreviated=False)
            print("MODEL_____memoty:", memory_summary)

            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            
            start_event.record()
            out, feaure1, feature2 = self.net.feature(img, keypoints)
            end_event.record()
            torch.cuda.synchronize()

            forward_time = start_event.elapsed_time(end_event)
            print(f"MODEL_____Forward time: {forward_time} milliseconds")
        else:
            out, feature1, feature2 = self.net.feature(0, img, keypoints)
        out = torch.sigmoid(out)
        out = out.detach().cpu().numpy().squeeze()
        feature1 = feature1.detach().cpu().numpy().squeeze()
        feature2 = feature2.detach().cpu().numpy().squeeze()

        return out, feature1, feature2

    
    def Framework_feature(self, dir):

        self.net.eval()
        for i, (img, keypoints, labels) in enumerate(tqdm(self.dataloader_val)):
            img = Variable(img.cuda())
            labels = Variable(labels.cuda())
            keypoints = Variable(keypoints.cuda())
            outs, feature1, feature2 = self.net.feature(img, keypoints)
            emb_pre = (torch.sigmoid(outs) > 0.5).cpu().detach().numpy()
            if i == 0:
                emb_pre_arr = emb_pre.copy()
            else:
                emb_pre_arr = np.vstack((emb_pre_arr, emb_pre))

            
            feature_pre1 = feature1.cpu().detach().numpy()
            if i == 0:
                feature_pre1_arr = feature_pre1.copy()
            else:
                feature_pre1_arr = np.concatenate((feature_pre1_arr, feature_pre1), axis=0)

            feature_pre2 = feature2.cpu().detach().numpy()
            if i == 0:
                feature_pre2_arr = feature_pre2.copy()
            else:
                feature_pre2_arr = np.concatenate((feature_pre2_arr, feature_pre2), axis=0)

            


            labels = labels.cpu().numpy()
            if i == 0:
                val_label_arr = labels.copy()
            else:
                val_label_arr = np.vstack((val_label_arr, labels))
            
            keypoints = keypoints.cpu().numpy()
            if i == 0:
                val_key_arr = keypoints.copy()
            else:
                val_key_arr = np.concatenate((val_key_arr, keypoints), axis=0)


        print(
            '***********************************************************************************************************')
        mA = self.log(0, val_label_arr, emb_pre_arr, 'emb')

        np.save('{}/feature1.npy'.format(dir), feature_pre1_arr)
        np.save('{}/feature2.npy'.format(dir), feature_pre2_arr)
        np.save('{}/label.npy'.format(dir), val_label_arr)
        np.save('{}/keys.npy'.format(dir), val_key_arr)
        np.save('{}/pre.npy'.format(dir), emb_pre_arr)
        return mA




