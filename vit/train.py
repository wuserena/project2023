import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
from timm.data.mixup import Mixup

from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate
from earlystopping import EarlyStopping

train_loss_list, train_acc_list = [], []
val_loss_list, val_acc_list = [], []


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    lowest_loss = 1.0

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     #transforms.RandAugment(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                     #transforms.RandomErasing()
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    mixup_fn = Mixup(mixup_alpha= 1.0,
                cutmix_alpha = 0.2,
                cutmix_minmax = None,
                prob = 1.0,
                switch_prob = 0.,
                mode = 'batch',
                label_smoothing = 0,
                num_classes = args.num_classes)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        checkpoint = weights_dict.get('pre_logits.fc.weight', None)
        if checkpoint is not None: #權重pre-logits=True
            del_keys = ['head.weight', 'head.bias']if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        else: #權重pre-logits=False
            del_keys = ['head.weight', 'head.bias']

        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    early_stopping = EarlyStopping(patience=args.patience, verbose=args.verbose)

    for epoch in range(args.epochs):
        # train
        '''
        for data in enumerate(train_dataset):
            img, label = data
            for img, label in train_loader:
                img, label =  mixup_fn(img, label)
            break
        '''
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        #torch.save(model.state_dict(), "./natural-weight/model-{}.pth".format(epoch))
        path = "./natural-weight/model-{}.pth".format(epoch)
        if val_loss > lowest_loss:
            lowest_loss = val_loss
            i = epoch
            best_model = model

        if args.early_stop:
            early_stopping(val_loss, model, save_path=path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            torch.save(model.state_dict(), path)

    torch.save(best_model.state_dict(), "./natural-weight/bestmodel-{}.pth".format(i))
    print("the best model:model-{}".format(i))

    x = range(len(train_loss_list))

    plt.figure()
    plt.plot(x, train_loss_list, label = 'train', color = 'royalblue')
    plt.plot(x, val_loss_list, label = 'valid', color = 'firebrick')
    plt.legend()
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.figure()
    plt.plot(x, train_acc_list, label = 'train', color = 'royalblue')
    plt.plot(x, val_acc_list, label = 'valid', color = 'firebrick')
    plt.legend()
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="D:/Serena/project/real_vs_fake_dataset/train")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='D:/Serena/project/vision_transformer/real-vs-fake/base/test1-bestmodel-49.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # Early Stopping
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--verbose', type=bool, default=False) #If True, prints a message for each validation loss improvement

    opt = parser.parse_args()

    main(opt)
