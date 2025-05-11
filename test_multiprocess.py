import os
import argparse
import torch
import numpy as np
import cv2
import time
from multiprocessing import Pool, cpu_count
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

def detect_face_single(img_path, args_dict):
    cfg = cfg_mnet if args_dict['network'] == 'mobile0.25' else cfg_re50
    net = RetinaFace(cfg=cfg, phase='test')
    device = torch.device("cpu" if args_dict['cpu'] else "cuda")

    pretrained_dict = torch.load(args_dict['trained_model'], map_location=device)
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    net.load_state_dict(pretrained_dict, strict=False)
    net.eval().to(device)

    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_raw is None:
        return

    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        loc, conf, landms = net(img)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance']) * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5).to(device)
        landms = (landms * scale1).cpu().numpy()

        inds = np.where(scores > args_dict['confidence_threshold'])[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args_dict['nms_threshold'])
        dets = dets[keep, :]
        landms = landms[keep]
        dets = np.concatenate((dets, landms), axis=1)

    rel_path = os.path.relpath(img_path, args_dict['dataset_folder'])
    txt_save_path = os.path.join(args_dict['save_folder'], 'result_test_mp_txt', rel_path.replace('.jpg', '.txt'))
    os.makedirs(os.path.dirname(txt_save_path), exist_ok=True)
    with open(txt_save_path, "w") as f:
        f.write(f"{rel_path}\n")
        f.write(f"{len(dets)}\n")
        for b in dets:
            x, y, x2, y2, conf = b[:5]
            w, h = int(x2 - x), int(y2 - y)
            f.write(f"{int(x)} {int(y)} {w} {h} {conf:.10f}\n")

    # Save image with landmarks
    if len(dets) > 0:
        for b in dets:
            if b[4] < args_dict['confidence_threshold']:
                continue
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            for j in range(5):
                cv2.circle(img_raw, (b[5 + j * 2], b[6 + j * 2]), 1, (0, 0, 255), 4)

        img_save_path = os.path.join(args_dict['save_folder'], 'result_test_mp_images', rel_path)
        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
        cv2.imwrite(img_save_path, img_raw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth')
    parser.add_argument('--network', default='mobile0.25')
    parser.add_argument('--save_folder', default='result_test/')
    parser.add_argument('--dataset_folder', default='data/widerface/test/images2/')
    parser.add_argument('--label_file', default='data/widerface/test/label2.txt')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--confidence_threshold', default=0.02, type=float)
    parser.add_argument('--nms_threshold', default=0.4, type=float)
    args = parser.parse_args()

    with open(args.label_file, 'r') as f:
        image_paths = [os.path.join(args.dataset_folder, line.strip()) for line in f if line.strip() and not line.startswith('#')]

    args_dict = vars(args)
    tasks = [(img_path, args_dict) for img_path in image_paths]

    start_time = time.time()

    with Pool(processes=cpu_count()) as pool:
        pool.starmap(detect_face_single, tasks)

    end_time = time.time()
    os.makedirs(args.save_folder, exist_ok=True)
    with open(os.path.join(args.save_folder, 'result_test_mp_time.txt'), 'w') as f:
        f.write(f"Start time: {start_time:.4f}\n")
        f.write(f"End time: {end_time:.4f}\n")
        f.write(f"Total time (s): {end_time - start_time:.2f}\n")

if __name__ == '__main__':
    main()
