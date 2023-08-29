import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.optim as optim

from torchvision import transforms
from timm.data.mixup import Mixup

from my_dataset import MyDataSet
from model import swin_large_patch4_window7_224_in22k as create_model
from utils import read_split_data, train_one_epoch, evaluate

train_loss_list, train_acc_list = [], []
val_loss_list, val_acc_list = [], []


def main(args):
    lowest_loss = 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     #transforms.RandomRotation(30),
                                     #transforms.RandAugment(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     #transforms.RandomErasing()
                                     ]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   #transforms.RandomAffine(degrees=(-30,30), translate=(0, 0.5), scale=(0.4, 0.5)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

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
    '''
    mixup_fn = Mixup(mixup_alpha= 1.0,
                    cutmix_alpha = 0.2,
                    cutmix_minmax = None,
                    prob = 1.0,
                    switch_prob = 0.,
                    mode = 'batch',
                    label_smoothing = 0,
                    num_classes = args.num_classes)
    '''
    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model_state_dict"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

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

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        torch.save({'model_state_dict':model.state_dict()}, "D:/Serena/project/natural sence/model-{}.pth".format(epoch))

        if val_loss < lowest_loss:
            lowest_loss = val_loss
            i = epoch
            best_model = model

    torch.save({'model_state_dict':best_model.state_dict()}, "D:/Serena/project/natural sence/bestmodel-{}.pth".format(i))
    print("the best model:model-{}".format(i))

    x = range(args.epochs)

    plt.figure()
    plt.plot(x, train_loss_list, label = 'train', color = 'g')
    plt.plot(x, val_loss_list, label = 'valid', color = 'b')
    plt.legend()
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.show()

    plt.figure()
    plt.plot(x, train_acc_list, label = 'train', color = 'g')
    plt.plot(x, val_acc_list, label = 'valid', color = 'b')
    plt.legend()
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="D:/Serena/project/natural scene_data set/seg_train/seg_train")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default="D:/Serena/project/swin_transformer/natural scene/test1-bestmodel-37.pth",
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()


    main(opt)
