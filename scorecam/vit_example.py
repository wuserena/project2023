import argparse
import cv2
import numpy as np
import torch

from score_cam import ScoreCAM
from grad_cam import GradCAM

from utils.image import show_cam_on_image, preprocess_image
from ablation_layer import AblationLayerVit
from vit_model import vit_large_patch16_224_in21k as create_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument(
        '--image-path',
        type=str,
        default='C:/Users/oscar/Downloads/016SSIRWSH.jpg',
        help='Input image path')
    parser.add_argument(
        '--weight_path',
        type=str,
        default="D:/Serena/project/vision_transformer/real-vs-fake/large/test5-early-model-43.pth",
        help='Input weight path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {
         "scorecam": ScoreCAM,
         "gradcam": GradCAM
         }

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model =  create_model(num_classes=args.num_classes, has_logits = False)
    model.load_state_dict(torch.load(args.weight_path, map_location="cpu"))

    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None #None: for eval labal, or int: for class indices

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite("C:/vscode/scorecam/image/vit_face_{}_cam.jpg".format(args.method), cam_image)
