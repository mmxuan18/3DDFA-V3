import argparse
import cv2
import os, time
import sys
import pickle
import logging
import torch
import numpy as np
from PIL import Image

from face_box import face_box
from model.recon import face_model
from util.preprocess import get_data_path
from util.io import visualize
import concurrent.futures as cf

def task_worker(args, video_name):
    if video_name.endswith(".mp4"):
        video_path = os.path.join(args.inputpath, video_name)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"open video filed path={video_path}")
            return
        
        start = time.time()
        logging.info(f"begin process={video_name}")
        recon_model = face_model(args)
        facebox_detector = face_box(args).detector

        video_prefix = video_name.replace(".mp4", "")
        gt_dir = os.path.join(args.savepath, "gt", video_prefix)
        lq_dir = os.path.join(args.savepath, "lq", video_prefix)
        mask_dir = os.path.join(args.savepath, "mask", video_prefix)
        kps_dir = os.path.join(args.savepath, "kps")

        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(lq_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(kps_dir, exist_ok=True)
        
        logging.info(f"process video={video_path} save={video_prefix}")
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        read_num = 0
        kps = []
        while True:
            ret, frame = cap.read()
            if not ret:
                if read_num != frame_num:
                    logging.error(f"video read frame failed path={video_path} frame={frame_num} read={read_num}")
                break
            read_num += 1

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            trans_params, im_tensor = facebox_detector(img)
            recon_model.input_img = im_tensor.to(args.device)
            results = recon_model.forward_pair()
            
            img_res = np.transpose(im_tensor.detach().cpu().numpy()[0], (1, 2, 0)) * 255
            img_res = img_res.astype(np.uint8)

            img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
            render_mask = results["render_mask"][0] * 255
            render_face = results["render_face"][0] * 255
            lm68 = results["ldm68"]
            kps.append(lm68)
            render_face = cv2.cvtColor(render_face.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(f"{gt_dir}/{read_num}.png", img_res.astype(np.uint8))
            cv2.imwrite(f"{lq_dir}/{read_num}.png", render_face)
            cv2.imwrite(f"{mask_dir}/{read_num}.png", render_mask.astype(np.uint8), cv2.GRAy)
            
            if read_num % 10 == 0:
                logging.info(f"process {video_path} {read_num}/{frame_num}")
        with open(f"{kps_dir}/{video_prefix}.pkl", "wb") as f:
            pickle.dump(kps, f)

        end = time.time()
        cost = end - start
        logging.info(f"finish process={video_name} timecost={cost}s")
    return video_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA-V3')

    parser.add_argument('-i', '--inputpath', default='examples/', type=str,
                        help='path to the test data, should be a image folder')
    parser.add_argument('-s', '--savepath', default='examples/results', type=str,
                        help='path to the output directory, where results (obj, png files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cuda or cpu' )

    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped and resized into (224,224,3).' )
    parser.add_argument('--detector', default='retinaface', type=str,
                        help='face detector for cropping image, support for mtcnn and retinaface')

    # save
    parser.add_argument('--ldm68', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 68 landmarks')
    parser.add_argument('--ldm106', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks')
    parser.add_argument('--ldm106_2d', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 106 landmarks, face profile is in 2d form')
    parser.add_argument('--ldm134', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show 134 landmarks' )
    parser.add_argument('--seg', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d without visible mask' )
    parser.add_argument('--seg_visible', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save and show segmentation in 2d with visible mask' )
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save obj use texture from BFM model')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='save obj use texture extracted from input image')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help='backbone for reconstruction, support for resnet50 and mbnetv3')
    parser.add_argument('--gen_feat', default="False", type=str, help='only gen feature')

    logging.basicConfig(filename='my_log.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    args = parser.parse_args()
    os.makedirs(args.savepath, exist_ok=True)
    if True:
        videos = []
        for item in os.listdir(args.inputpath):
            video_name = item.strip()
            if not video_name.endswith(".mp4"):
                continue
            videos.append(video_name)
        
        with cf.ThreadPoolExecutor(max_workers=8) as executor:
            # 将任务提交给线程池，返回futures列表
            futures = [executor.submit(task_worker, args, video) for video in videos]

            # 等待所有任务完成，并获取结果
            for future in cf.as_completed(futures):
                result = future.result()
                print(f"Task returned: {result}")