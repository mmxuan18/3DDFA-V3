import os,sys
import cv2
import pickle, numpy
from mtcnn import MTCNN
import matplotlib.pyplot as plt

def detect_faces_and_keypoints(image_path, detector):
    # 读取图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用MTCNN检测人脸关键点
    faces = detector.detect_faces(image)

    # 在图像上绘制人脸关键点
    pps = []
    for face in faces:
        keypoints = face['keypoints']
        # for key, point in keypoints.items():
        #     x, y = point
        #     cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
        pps.append(keypoints['left_eye'])
        pps.append(keypoints['right_eye'])
        pps.append(keypoints['nose'])
        pps.append(keypoints['mouth_left'])
        pps.append(keypoints['mouth_right'])
        
        
        break

    # 显示图像
    # plt.imshow(image)
    # plt.show()
    return pps

if __name__ == "__main__":
    root_dir = "/root/mlinxiang/vh_exp/FaceFormer/HDTF/data_fps25/"
    a = []
    for video in os.listdir(root_dir):
        if  not video.endswith(".mp4"):
            continue
        
        cap = cv2.VideoCapture(os.path.join(root_dir, video))
        print(f"{video}={cap.get(cv2.CAP_PROP_FRAME_COUNT)} / {cap.get(cv2.CAP_PROP_FPS)}")
        a.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    print(sorted(a))
    
    sys.exit()
    with open("/root/mlinxiang/vh_exp/FaceFormer/HDTF/npy/RD_Radio10_000.npy", "rb") as f:
        a = pickle.load(f)
        tt = []
        for item in a:
            tt.append(item['v3d'])
        
        numpy.save("a.npy", numpy.array(tt))
        sys.exit()
    
    
    
    
    # 设置图像目录
    image_dir = '/root/mlinxiang/vh_exp/3DDFA-V3/examples/video/'
    dst_dir = "/root/mlinxiang/vh_exp/3DDFA-V3/examples/video/detections/"

    # 创建MTCNN对象
    detector = MTCNN()

    # 遍历目录下的所有图片
    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, file_name)
            print(f"Processing {image_path}")
            pps = detect_faces_and_keypoints(image_path, detector)
            
            name,_ = os.path.splitext(file_name)
            d_name = os.path.join(dst_dir, name, ".txt")
            with open(d_name, "w") as f:
                for p in pps:
                    print(f"{p[0]} {p[1]}", file=f)