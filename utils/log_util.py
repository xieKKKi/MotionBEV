# -*- coding:utf-8 -*-
# author: Jiapeng Xie
# @file: log_util.py 

import os
import logging
import datetime
import shutil


def make_log_dir(arch_cfg, data_cfg, name=None, model_save_path="./model_save_dir"):
    model_save_path = model_save_path + '/' + datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + "-" + name
    # create log folder
    try:
        if os.path.isdir(model_save_path):
            if os.listdir(model_save_path):
                print("Log Directory is not empty. Remove. ")
                shutil.rmtree(model_save_path)
        os.makedirs(model_save_path)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print("\033[32m Copying files to %s for further reference.\033[0m" % model_save_path)
        shutil.copyfile(arch_cfg, model_save_path + "/arch_cfg.yaml")
        shutil.copyfile(data_cfg, model_save_path + "/data_cfg.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()
    return model_save_path


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def save_to_log(logdir, logfile, message):
    f = open(logdir + '/' + logfile, "a")
    f.write(message + '\n')
    f.close()
    return