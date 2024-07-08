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

def main(args):

    recon_model = face_model(args)
    facebox_detector = face_box(args).detector
    im_path = get_data_path(args.inputpath)

    for i in range(len(im_path)):
        print(i, im_path[i])
        im = Image.open(im_path[i]).convert('RGB')
        trans_params, im_tensor = facebox_detector(im)

        recon_model.input_img = im_tensor.to(args.device)
        results = recon_model.forward()

        if not os.path.exists(os.path.join(args.savepath, im_path[i].split('/')[-1].replace('.png','').replace('.jpg',''))):
            os.makedirs(os.path.join(args.savepath, im_path[i].split('/')[-1].replace('.png','').replace('.jpg','')))
        my_visualize = visualize(results, args)

        my_visualize.visualize_and_output(trans_params, cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), \
            os.path.join(args.savepath, im_path[i].split('/')[-1].replace('.png','').replace('.jpg','')), \
            im_path[i].split('/')[-1].replace('.png','').replace('.jpg',''))
        # my_visualize.visualize_and_output(trans_params, cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), \
        #     os.path.join(args.savepath), \
        #     im_path[i].split('/')[-1].replace('.png','').replace('.jpg',''))

def gen_feature(args):
    recon_model = face_model(args)
    facebox_detector = face_box(args).detector
    
    for item in os.listdir(args.inputpath):
        video_name = item.strip()
        if not video_name.endswith(".mp4"):
            continue

        video_path = os.path.join(args.inputpath, video_name)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"open video filed path={video_path}")
            continue
        
        video_feats_name = video_name.replace(".mp4", ".npy")
        video_feats_path = os.path.join(args.savepath, video_feats_name)
        
        logging.info(f"process video={video_path} save={video_feats_path}")
        frame_feats = []
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        read_num = 0
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
            results = recon_model.forward(is_gen_feature=True)
            results['trans_param'] = trans_params
            frame_feats.append(results)
            if read_num % 10 == 0:
                logging.info(f"process {video_path} {read_num}/{frame_num}")
            
            # if not os.path.exists(os.path.join(args.savepath, str(read_num))):
            #     os.makedirs(os.path.join(args.savepath, str(read_num)))
            # my_visualize = visualize(results, args)

            # my_visualize.visualize_and_output(trans_params, frame, \
            #     os.path.join(args.savepath, str(read_num)), \
            #     str(read_num))
            # if read_num > 10:
            #     break
        
        with open(video_feats_path, "wb") as f:
            pickle.dump(frame_feats, f)
            
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
        video_feats_name = video_name.replace(".mp4", ".npy")
        video_feats_path = os.path.join(args.savepath, video_feats_name)
        
        logging.info(f"process video={video_path} save={video_feats_path}")
        frame_feats = []
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        read_num = 0
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
            results = recon_model.forward(is_gen_feature=True)
            results['trans_param'] = trans_params
            frame_feats.append(results)
            if read_num % 10 == 0:
                logging.info(f"process {video_path} {read_num}/{frame_num}")
            
            # if not os.path.exists(os.path.join(args.savepath, video_name, str(read_num))):
            #     os.makedirs(os.path.join(args.savepath, video_name, str(read_num)))
            # my_visualize = visualize(results, args)

            # my_visualize.visualize_and_output(trans_params, frame, \
            #     os.path.join(args.savepath, video_name, str(read_num)), \
            #     str(read_num))
            # if read_num > 2:
            #     break
        
        with open(video_feats_path, "wb") as f:
            pickle.dump(frame_feats, f)
        
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
    if args.gen_feat == "True":
        videos = []
        for item in os.listdir(args.inputpath):
            video_name = item.strip()
            if not video_name.endswith(".mp4"):
                continue
            videos.append(video_name)
        
        with cf.ThreadPoolExecutor(max_workers=1) as executor:
            # 将任务提交给线程池，返回futures列表
            futures = [executor.submit(task_worker, args, video) for video in videos]

            # 等待所有任务完成，并获取结果
            for future in cf.as_completed(futures):
                result = future.result()
                print(f"Task returned: {result}")
            
            
        # gen_feature(args)
    else:
        main(args)
