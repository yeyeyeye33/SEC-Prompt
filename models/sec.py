# watermark version

import logging
import os
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from backbone.vpt_backbone import SimpleVitNet
from models.base import BaseLearner

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 1


def cos_loss(cosine, label):
    loss = 0
    for i, y in enumerate(label):
        loss += 1 - cosine[i, y]
    return loss / len(label)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.fs_lr = args['fs_lr']
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.midfeature = None

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):
        # use class prototype as classifier weights.
        model = model.eval()
        embedding_list = []
        label_list = []
        if self._cur_task > 0:
            with torch.no_grad():
                for i, batch in enumerate(trainloader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model(data)['features']
                    embedding_list.append(embedding.cpu())
                    label = label.repeat(int(embedding.shape[0] / label.shape[0]))
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
        else:
            with torch.no_grad():
                for i, batch in enumerate(trainloader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model(data)['features']
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto

        return model


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # print(self._total_classes)
        self._network.update_fc(self._total_classes)
        self._network.backbone.TSP.process_task_count(self._total_classes)
        self._network.backbone.RSP.process_task_count()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if isinstance(self.args['kshot'], int) and self._known_classes > 0:
            train_bs = self.args['fs_batch_size']
        else:
            train_bs = self.batch_size
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", kshot=self.args["kshot"])
        self.train_dataset = train_dataset

        self.train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers)

        self.data_manager = data_manager

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        test_curr_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test",
                                                     mode="test")
        self.test_curr_loader = DataLoader(test_curr_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=num_workers)
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", kshot=self.args["kshot"])

        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=int(self.batch_size / 2),
                                                    shuffle=False, num_workers=num_workers)

        logging.info("training set size: {}, fc construct set size: {}".format(len(train_dataset),
                                                                               len(train_dataset_for_protonet)))

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):

        self._network.to(self._device)
        if self._cur_task > 0:
            self._network.backbone.Freeze_new()
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info('total parameters: {}'.format(total_params))
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info('trainable parameters: {}'.format(total_trainable_params))

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

        if os.path.exists(self.args["base_model_path"]) and self._cur_task == 0:
            logging.info(
                '================= load base model from: {} ================='.format(self.args["base_model_path"]))
            print(self._cur_task)
            self._network.load_state_dict(torch.load(self.args["base_model_path"]))
            # self.replace_midfeature(train_loader_for_protonet, self._network, None)
            # self.replace_fc(train_loader_for_protonet, self._network, None)

        else:
            if self._cur_task > 0:
                self.replace_fc(train_loader_for_protonet, self._network, None)

            if self._cur_task == 0:
                if self.args['optimizer'] == 'sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                          weight_decay=self.weight_decay)
                elif self.args['optimizer'] == 'adam':
                    optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'],
                                                                 eta_min=self.min_lr)
                self._init_train(train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler)
            else:
                if self.args['optimizer'] == 'sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.fs_lr,
                                          weight_decay=self.weight_decay)
                elif self.args['optimizer'] == 'adam':
                    optimizer = optim.AdamW(self._network.parameters(), lr=self.fs_lr, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['fs_epoch'],
                                                                 eta_min=self.min_lr)
                self._init_train(train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler)
            self.replace_fc(train_loader_for_protonet, self._network, None)
            # self.replace_midfeature(train_loader_for_protonet, self._network, None)
            if self._cur_task == 0:
                torch.save(self._network.state_dict(), self.args["base_model_path"])

    def eval_task(self):
        y_pred, y_true = self._eval_acc(self.test_loader)
        accy = self._evaluate(y_pred, y_true)
        return accy

    def _eval_acc(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        all_outputs, all_embedding = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                out = self._network(inputs)
                outputs = out["logits"]
                embedding = out["features"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu())
            all_embedding.append(embedding.cpu())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        all_outputs = torch.cat(all_outputs)
        all_embedding = torch.cat(all_embedding)

        return y_pred, y_true  # [N, topk]

    # naive train
    def _init_train(self, train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler):
        if isinstance(self.args['kshot'], int) and self._known_classes > 0:
            total_epoch = self.args['fs_epoch']
        else:
            total_epoch = self.args['tuned_epoch']
        for _, epoch in enumerate(range(total_epoch)):
            self._network.train()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # print(inputs.shape)
                if self._cur_task == 0:

                    out = self._network(inputs, targets=targets)
                    logits = out["logits"]
                    loss_pc = out['loss_match']
                    loss = F.cross_entropy(logits, targets) + loss_pc * self.args["beta"]

                else:

                    out = self._network(inputs, train=True, targets=targets)
                    logits = out["logits"]
                    loss_pc = out['loss_match']
                    targets = targets.repeat(int(logits.shape[0] / targets.shape[0]))
                    loss = F.cross_entropy(logits, targets) + loss_pc * self.args["beta"]
                    # print(loss_pc)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            scheduler.step()
