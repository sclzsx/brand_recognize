from pathlib import Path
import cv2
import os
import random
import shutil

def mkdir(save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

def save_video_frames(video_root, image_root, max_cnt=2000):
    for video_path in Path(video_root).rglob('*.*'):
        name = video_path.name.split('.')[0]
        save_dir = image_root + '/' + name
        mkdir(save_dir)

        print(name)
        videoCapture = cv2.VideoCapture(str(video_path))
        cnt = 0
        while True:
            success, frame = videoCapture.read()
            if (not success) or cnt > max_cnt:
                break
            cv2.imwrite(save_dir + '/' + str(cnt) + '.jpg', frame)
            cnt += 1
        videoCapture.release()
        print('saved {} frames'.format(cnt))

def split_train_test(image_root, val_rate=0.2, test_rate=0.2):
    image_dirs = [i for i in Path(image_root).iterdir() if i.is_dir()]
    for image_dir in image_dirs:
        image_paths = [i for i in Path(image_dir).glob('*.jpg')]
        random.shuffle(image_paths)
        val_num = int(len(image_paths) * val_rate)
        test_num = int(len(image_paths) * test_rate)
        for _ in range(test_num):
            image_path = image_paths.pop()
            save_path = str(image_path).replace('images', 'test')
            mkdir(str(Path(save_path).parent))
            shutil.move(str(image_path), save_path)
        for _ in range(val_num):
            image_path = image_paths.pop()
            save_path = str(image_path).replace('images', 'val')
            mkdir(str(Path(save_path).parent))
            shutil.move(str(image_path), save_path)
        for _ in range(len(image_paths)):
            image_path = image_paths.pop()
            save_path = str(image_path).replace('images', 'train')
            mkdir(str(Path(save_path).parent))
            shutil.move(str(image_path), save_path)
    shutil.rmtree(image_root)



if __name__ == '__main__':
    video_root = 'data/Camera Roll'
    image_root = 'data/images'

    save_video_frames(video_root, image_root)
    split_train_test(image_root)