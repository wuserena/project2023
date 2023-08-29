import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from gradcam import heatmap
from utils_heatmap import show_cam_on_image
from model import swin_base_patch4_window7_224_in22k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "C:/Users/oscar/Downloads/016SSIRWSH.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]qq
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = 'C:/vscode/real-vs-fake_class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    num_classes = 2
    model = create_model(num_classes).to(device)
    # load model weights
    model_weight_path = "D:/Serena/project/swin_transformer/real-vs-fake/base-test10-bestmodel-21.pth"
    #model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.load_state_dict(torch.load(model_weight_path, map_location="cpu")["model_state_dict"], strict=False)

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.figure()


    input_tensor, cam, img_cam = heatmap(create_model, num_classes, img_path, model_weight_path)
    grayscale_cam = cam(input_tensor=input_tensor, target_category= 1) #int(predict_cla)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_cam / 255., grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.show()



if __name__ == '__main__':
    main()
