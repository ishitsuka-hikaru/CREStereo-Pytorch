import torch
import torch.nn.functional as F
import numpy as np
import cv2
# from imread_from_url import imread_from_url
import argparse
import time

from nets import Model

device = 'cuda'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):

        # print("Model Forwarding...")
        imgL = left.transpose(2, 0, 1)
        imgR = right.transpose(2, 0, 1)
        imgL = np.ascontiguousarray(imgL[None, :, :, :])
        imgR = np.ascontiguousarray(imgR[None, :, :, :])

        imgL = torch.tensor(imgL.astype("float32")).to(device)
        imgR = torch.tensor(imgR.astype("float32")).to(device)

        imgL_dw2 = F.interpolate(
                imgL,
                size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
        )
        imgR_dw2 = F.interpolate(
                imgR,
                size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
        )
        # print(imgR_dw2.shape)
        with torch.inference_mode():
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

                pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
        pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

        return pred_disp


def get_opts():
        p = argparse.ArgumentParser()
        p.add_argument('--model_path', type=str, default='models/crestereo_eth3d.pth')
        p.add_argument('--left', type=str)
        p.add_argument('--right', type=str)
        p.add_argument('--size_h', type=int, default=1536)
        p.add_argument('--size_w', type=int, default=1024)
        p.add_argument('--n_iter', type=int, default=20)
        p.add_argument('--output', type=str, default=None)
        p.add_argument('--meas', type=str, default=None)
        
        return p.parse_args()

        
if __name__ == '__main__':
        t_begin = time.time()
        
        opt = get_opts()

        # left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
        # right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")
        left_img = cv2.imread(opt.left)  # cv2.imread('../shared/data/test/left.png')
        right_img = cv2.imread(opt.right)  # cv2.imread('../shared/data/test/right.png')

        in_h, in_w = left_img.shape[:2]
        if opt.size_h:
                in_h = opt.size_h
        if opt.size_w:
                in_w = opt.size_w

        # Resize image in case the GPU memory overflows
        eval_h, eval_w = (in_h,in_w)
        assert eval_h%8 == 0, "input height should be divisible by 8"
        assert eval_w%8 == 0, "input width should be divisible by 8"
        
        imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

        model = Model(max_disp=256, mixed_precision=False, test_mode=True)
        model.load_state_dict(torch.load(opt.model_path), strict=True)
        model.to(device)
        model.eval()

        # meas pre
        if opt.meas:
                t_pre = time.time() - t_begin
                print(f't_pre={t_pre:5.3f}', end=', ')
                with open(opt.meas + '_pre.txt', 'a') as f:
                        f.write(f'{t_pre}\n')
                        
        pred = inference(imgL, imgR, model, n_iter=opt.n_iter)

        # meas infer
        if opt.meas:
                t_infer = time.time() - t_pre - t_begin
                print(f't_infer={t_infer:5.3f}', end=', ')
                with open(opt.meas + '_infer.txt', 'a') as f:
                        f.write(f'{t_infer}\n')


        t = float(in_w) / float(eval_w)
        disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

        # combined_img = np.hstack((left_img, disp_vis))
        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        # cv2.imshow("output", combined_img)
        # cv2.imwrite("output.jpg", disp_vis)
        cv2.imwrite(opt.output, disp_vis)
        # cv2.waitKey(0)

        # meas post
        if opt.meas:
                t_post = time.time() - t_infer - t_pre - t_begin
                print(f't_post={t_post:5.3f}')
                with open(opt.meas + '_post.txt', 'a') as f:
                        f.write(f'{t_post}\n')
