import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.utils.data as Data
from utils.binary import assd
from distutils.version import LooseVersion

from Datasets.skin_inter import inter_dataset
from utils.transform import ISIC2018_transform

from Models.networks.network import Comprehensive_Atten_Unet

from utils.dice_loss import get_soft_label, val_dice_isic
from utils.dice_loss import Intersection_over_Union_isic

from time import *
from PIL import Image

Test_Model = {'Comp_Atten_Unet': Comprehensive_Atten_Unet}


def test_isic(test_loader, model, args):
    isic_dice = []
    isic_iou = []
    isic_assd = []
    infer_time = []

    model.eval()
    print(args.root_path, args.save_path)
    for step, (img, lab, imgpath) in enumerate(test_loader):
        image = img.float().cuda()
        target = lab.float().cuda()

        begin_time = time()
        output,centerf = model(image)
        end_time = time()
        pred_time = end_time - begin_time
        infer_time.append(pred_time)

        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_soft = get_soft_label(output_dis, args.num_classes)
        target_soft = get_soft_label(target, args.num_classes)  # get soft label

        label_arr = target_soft.cpu().numpy().astype(np.uint8)
        output_arr = output_soft.cpu().byte().numpy().astype(np.uint8)
        binary_mask = np.argmax(output_arr[0],axis=-1)
        binary_image = (binary_mask * 255).astype(np.uint8)
        originimg = Image.open(imgpath[0])
        image = Image.fromarray(binary_image).resize(originimg.size)
        maskpath = imgpath[0].replace(args.root_path, args.save_path)
        mask_dir = os.path.dirname(maskpath)
        name = os.path.basename(maskpath)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        print('imgpath',imgpath)
        image.save(os.path.join(mask_dir,name.replace('.jpeg','.png')))

       

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='U-net add Attention mechanism for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='Comp_Atten_Unet',
                        help='a name for identitying the model. Choose from the following options: Unet_fetus')
    # Path related arguments
    parser.add_argument('--root_path', default='./ZCH-Data-test',
                        help='root directory of data')
    parser.add_argument('--save_path', default='./ZCH-Data-test-seg',
                        help='save directory of data')
    parser.add_argument('--suffix', default='.jpg',
                        help='suffix of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--save', default='./result',
                        help='folder to outoput result')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--epoch', type=int, default=300, metavar='N',
                        help='choose the specific epoch checkpoints')

    # other arguments
    parser.add_argument('--out_size', default=(224, 300), help='the output image size')
    parser.add_argument('--att_pos', default='dec', type=str,
                        help='where attention to plug in (enc, dec, enc\&dec)')
    parser.add_argument('--view', default='axial', type=str,
                        help='use what views data to test (for fetal MRI)')
    parser.add_argument('--val_folder', default='folder0', type=str,
                        help='which cross validation folder')

    args = parser.parse_args()
    args.ckpt = os.path.join(args.ckpt, args.val_folder, args.id)


    # loading the dataset
    print('loading the {0} dataset ...'.format('test'))
    testset = inter_dataset(dataset_folder=args.root_path, suffix=args.suffix, train_type='test', transform=ISIC2018_transform)
    testloader = Data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
    print('Loading is done\n')

    # Define model
    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to train the network')
        args.num_input = 3
        args.num_classes = 2
        args.out_size = (224, 300)
        model = Test_Model[args.id](args, args.num_input, args.num_classes).cuda()

    # Load the trained best model
    modelname = args.ckpt + '/' + 'min_loss_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
        test_isic(testloader, model, args)
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    
