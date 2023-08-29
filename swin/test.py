import os

import torch
from torchvision import transforms
from my_dataset import MyDataSet

from utils import read_test_data, evaluate
from model import swin_base_patch4_window7_224_in22k as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = "C:/vscode/data/face_test/test"
    img_size = 224

    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_images_path, test_images_label = read_test_data(data_path)

    test_dataset = MyDataSet(images_path=test_images_path,
                              images_class=test_images_label,
                              transform=data_transform)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=test_dataset.collate_fn)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "D:/Serena/project/real-vs-fake/base-test8-bestmodel-40.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location="cpu")["model_state_dict"], strict=False)

    test_loss, test_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=1,
                                     valid=False)
    print('loss = {}'.format(test_loss))
    print('accuracy = {}'.format(test_acc))

if __name__ == '__main__':
    main()
