import glob
import logging
import os
import shutil
import pickle
from os.path import join, dirname, exists

import cv2
import gym
from gym.wrappers import Monitor
from skimage import img_as_ubyte
import imageio

def log(msg, verbose=False):
    if verbose: print(msg)
    logging.info(msg)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))

def pickle_save(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def create_video(frame_dir, video_dir, agent_hl, size, length, fps):
    img_array = []
    for i in range(length):
        img = cv2.imread(os.path.join(frame_dir, agent_hl + f'_Frame{i}.png'))
        img_array.append(img)
    out = cv2.VideoWriter(os.path.join(video_dir, agent_hl) + '.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def save_image(path, name, img):
    imageio.imsave(path + '/' + name + '.png', img_as_ubyte(img))


def clean_dir(path, file_type='', hard=False):
    if not hard:
        files = glob.glob(path + "/*" + file_type)
        for f in files:
            os.remove(f)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def make_clean_dirs(path, no_clean=False, file_type='', hard=False):
    try:
        os.makedirs(path)
    except:  # if exists
        if not no_clean: clean_dir(path, file_type, hard)


def video_schedule(config, videos):
    # linear capture schedule
    return (lambda e: True) if videos else \
        (lambda e: videos and (e == config.num_episodes - 1 or
                               e % int(config.num_episodes / config.num_recorded_videos) == 0))


def mark_agent(img, position=None, color=255, thickness=2):
    img2 = img.copy()
    top_left = (position[0], position[1])
    bottom_right = (position[0] + 30, position[1] + 30)
    cv2.rectangle(img2, top_left, bottom_right, color, thickness)
    return img2
