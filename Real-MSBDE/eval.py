import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.measure import compare_psnr
import time
import cv2


def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            tm = time.time()

            pred = model(input_img)[2]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().permute(1, 2, 0).numpy()
            label_numpy = label_img.squeeze(0).cpu().permute(1, 2, 0).numpy()

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                im = np.uint16(pred_numpy * 65535.0)
                im_HBD = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_name, im_HBD)
           
            psnr = compare_psnr(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print("Average time: %f" % adder.average())
