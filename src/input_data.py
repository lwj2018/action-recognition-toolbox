import os
import cv2 as cv
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

class DataReader:
  def __init__(self, filename = None, shuffle = True, mode = "train", mirror = True):
    assert filename is not None, "\033[31m filename of DataReader can not be None \033[0m"
    self._total_index = 0
    self._file_list = []
    self._clipname_list = []
    self._start_frm_list = []
    self._label_list = []
    self._filename = filename
    lines = open(filename,'r')
    lines = list(lines)
    for line in lines:
      line = line.strip('\n').split()
      temp_filename = line[0]
      clipname = line[0].split('/')[-1]
      start_frm = int(line[1])
      temp_label = int(line[2])
      self._file_list.append(temp_filename)
      self._clipname_list.append(clipname)
      self._start_frm_list.append(start_frm)
      self._label_list.append(temp_label)
    self._shuffle_index = np.arange(len(lines))
    self._shuffle = shuffle
    self._mirror = mirror 
    if mode == "test":
      self._shuffle = False
      self._mirror = False
    self._chunk_size = len(lines)
    if self._shuffle == True:
      np.random.shuffle(self._shuffle_index)
    #self._np_mean = np.load('crop_mean.npy').reshape([16, 128, 171, 3]) 
    self._mode = mode
    return

  def get_frames_data(self, filename=None, start_frame=None, np_mean=None, num_frames_per_clip=16):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    crop_size = 112
    width = 171
    height = 128
    x_maxoff = width - crop_size
    y_maxoff = height - crop_size
    ret_arr = []
    for i in range(start_frame, start_frame + num_frames_per_clip):
      image_name = '/home/liweijie/C3D/C3D-v1.0/examples/c3d_finetuning/'+str(filename) + '/' + '%06d.jpg' % i
      img = Image.open(image_name)
      img = np.array(img)
      #img = img - np_mean[i - start_frame]
      if self._mode == "train":
        x_off = np.random.randint(0, x_maxoff)
        y_off = np.random.randint(0, y_maxoff)
      elif self._mode == "test":
        x_off = int((width - crop_size)/2)
        y_off = int((height - crop_size)/2)
      img = img[y_off : y_off+crop_size, x_off : x_off+crop_size, :]
      if (self._mode == "train") and (self._mirror == True) and (np.random.randint(2) == 0):
        img = img[:,::-1,:]
      ret_arr.append(img)
    return ret_arr

  def read_clip_and_label(self, batch_size, num_frames_per_clip=16, crop_size=112):
    data = []
    label = []
    for i in range(batch_size):     
      # prevent from sample_index out of list , restart from the head
      if self._total_index >= self._chunk_size:
        self._total_index = 0 
        if self._shuffle == True:
          np.random.shuffle(self._shuffle_index)
      # get a list of images 
      now_index = self._shuffle_index[self._total_index]
      dirname = self._file_list[now_index]
      start_frame = self._start_frm_list[now_index]
      tmp_label = self._label_list[now_index]
      clipname = self._clipname_list[now_index]
      # DEBUG
      if(i == 0):
        print("read clip from: {}  {}".format(dirname, start_frame))
      tmp_data = self.get_frames_data(filename=dirname, start_frame=start_frame) # tmp_data is a list of images, length = 16
      frames = np.array(tmp_data)
      if(len(tmp_data)!=0):  
        data.append(frames)
        label.append(tmp_label)
        self._total_index = self._total_index + 1
      else:
        print("\033[0;31m can not read the image sequence correctly \033[0m")

    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_data, np_arr_label, clipname

    