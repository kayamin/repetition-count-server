'''
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
'''

import pickle
import time
import numpy
import cv2
from scipy.ndimage import filters
import theano
import scipy
import theano.tensor as T
import sys, getopt

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import pdb

sys.path.append('../')
from layers import RepetitionCountingNet
from util.Dataset_loader import Dataset_loader
from util.log_learning import log_learning

class state:
    NO_REP = 1
    IN_REP = 2
    COOLDOWN = 3

# global vars
detector_strides = [5,7,9]
static_th = 10
norep_std_th = 13
norep_ent_th = 1.0
inrep_std_th = 13
inrep_ent_th = 1.1
lastsix_ent_th = 1.1
history_num = 9

in_time = 0
out_time = 0
cooldown_in_time = 0
cooldown_out_time = 0
frame_rate = 30
num_of_vids = 25

class RepDetector:

    def __init__(self, D_model, proFrame, st_number):
        self.D_model = D_model
        self.stride_number = st_number
        self.frame_set = numpy.zeros([20,proFrame.shape[0],proFrame.shape[1]])
        for i in range(20):
            self.frame_set[i,:,:] = proFrame

        self.rep_count = 0
        self.frame_residue = 0
        self.st_entropy = 0
        self.st_std = 0
        self.std_arr = numpy.zeros(history_num)
        self.ent_arr = numpy.zeros(history_num)+2
        self.label_array = numpy.zeros(history_num)
        self.count_array = numpy.zeros(history_num)
        self.cur_std = 0
        self.cur_entropy = 0


    # receive a list of 20 frames and return the ROI scaled to 50x50
    def get_boundingbox(self):

        fstd = numpy.std(self.frame_set,axis=0)
        framesstd = numpy.mean(fstd)
        th = framesstd
        ones = numpy.ones(10)
        big_var = (fstd>th)


        if (framesstd==0): # no bb, take full frame
            frameROIRes = numpy.zeros([20,50,50])
            for i in range(20):
                frameROIRes[i,:,:] = scipy.misc.imresize(self.frame_set[i,:,:], size=(50,50),interp='bilinear')

            frameROIRes = numpy.reshape(frameROIRes, (1,frameROIRes.shape[0]*frameROIRes.shape[1]*frameROIRes.shape[2]))
            frameROIRes = frameROIRes.astype(numpy.float32)

            return (frameROIRes)

        big_var = big_var.astype(numpy.float32)
        big_var = filters.convolve1d(big_var, ones, axis=0)
        big_var = filters.convolve1d(big_var, ones, axis=1)

        th2 = 80
        i,j = numpy.nonzero(big_var>th2)

        if (i.size > 0):

            si = numpy.sort(i)
            sj = numpy.sort(j)


            ll = si.shape[0]
            th1 = round(ll*0.02)
            th2 = int(numpy.floor(ll*0.98))
            y1 = si[th1]
            y2 = si[th2]
            x1 = sj[th1]
            x2 = sj[th2]
            # cut image ROI
            if (((x2-x1)>0) and ((y2-y1)>0)):
                framesRoi = self.frame_set[:,y1:y2,x1:x2]
            else:
                framesRoi = self.frame_set[:,:,:]
        else:
            framesRoi = self.frame_set[:,:,:]

        # resize to 50x50
        frameROIRes = numpy.zeros([20,50,50])
        for i in range(20):
            frameROIRes[i,:,:] = scipy.misc.imresize(framesRoi[i,:,:], size=(50,50),interp='bilinear')

        riofstd = numpy.std(frameROIRes,axis=0)
        self.cur_std = numpy.mean(riofstd)

        # return 4d ndarray
        frameROIRes = frameROIRes.reshape(-1, 20, 50, 50).astype(numpy.float32)

        return (frameROIRes)


    def do_local_count(self, initial):

        framesArr = self.get_boundingbox()
        framesArr = Variable(torch.FloatTensor(framesArr))
        # classify
        output_label , pYgivenX  = self.D_model.get_output_labels(framesArr)
        self.cur_entropy = - (pYgivenX.data*numpy.log(pYgivenX.data)).sum()
        # count
        output_label = output_label[0] + 3
        print(output_label.data[0])
        print(self.cur_entropy)
        self.label_array = numpy.delete(self.label_array,0,axis=0)
        self.label_array = numpy.insert(self.label_array, history_num-1 , output_label.data[0], axis=0)
        #take median of the last frames
        med_out_label = numpy.ceil(numpy.median(self.label_array[history_num-4:history_num]))
        med_out_label = med_out_label.astype('int32')

        if initial:
            self.rep_count = 20 / (med_out_label)
            self.frame_residue = 20 % (med_out_label)
        else:
            self.frame_residue += 1
            if (self.frame_residue >= med_out_label):
                self.rep_count += 1;
                self.frame_residue = 0;



    def count(self, proFrame):

        # globals
        global in_time,	out_time, cooldown_in_time, cooldown_out_time
        global global_counter, winner_stride, cur_state, in_frame_num, actions_counter

        # insert new frame
        self.frame_set = numpy.delete(self.frame_set,0,axis=0)
        self.frame_set = numpy.insert(self.frame_set, 19 , proFrame, axis=0)

        if (cur_state == state.NO_REP):
            self.do_local_count(True)
        if ((cur_state == state.IN_REP) and (winner_stride == self.stride_number)):
            self.do_local_count(False)
        if (cur_state == state.COOLDOWN):
            self.do_local_count(True)

        # common to all states
        if (self.cur_std < static_th):
            self.cur_entropy = 2

        self.count_array = numpy.delete(self.count_array,0,axis=0)
        self.count_array = numpy.insert(self.count_array, history_num-1 , self.rep_count, axis=0)
        self.ent_arr = numpy.delete(self.ent_arr,0,axis=0)
        self.ent_arr = numpy.insert(self.ent_arr, history_num-1 , self.cur_entropy, axis=0)
        self.std_arr = numpy.delete(self.std_arr,0,axis=0)
        self.std_arr = numpy.insert(self.std_arr, history_num-1 , self.cur_std, axis=0)
        self.st_std = numpy.median(self.std_arr)
        self.st_entropy = numpy.median(self.ent_arr)

        if (cur_state == state.NO_REP):
            # if we see good condition for rep take the counting and move to rep state
            if ((self.st_std > norep_std_th) and (self.st_entropy < norep_ent_th)):
                # start counting!
                actions_counter += 1
                cur_state = state.IN_REP
                global_counter = self.rep_count
                winner_stride = self.stride_number
                in_time = in_frame_num/30
        if ((cur_state == state.IN_REP) and (winner_stride == self.stride_number)):
            lastSixSorted = numpy.sort(self.ent_arr[history_num-8:history_num])
            # if we see good condition for rep take the counting and move to rep state
            # also, if there were 2 below th in the last entropies, don't stop.
            if (((self.st_std > inrep_std_th) and (self.st_entropy < inrep_ent_th)) or  (lastSixSorted[1] < lastsix_ent_th)):
                # continue counting
                global_counter = self.rep_count
            else:
                out_time = in_frame_num/30
                if (((out_time-in_time)<4) or (self.rep_count<5)):
                    # fast recovery mechnism, start over
                    actions_counter -= 1
                    global_counter = 0
                    cur_state = state.NO_REP
                    print('fast recovery applied !!')
                else:
                    # rewind redundant count mechanism
                    # find how many frames pass since we have low entropy
                    frames_pass = 0
                    reversed_ent = self.ent_arr[::-1]
                    for cent in reversed_ent:
                        if (cent > inrep_ent_th):
                            frames_pass += 1
                        else:
                            break
                    # calc if and how many global count to rewind
                    reversed_cnt = self.count_array[::-1]
                    frames_pass = min(frames_pass, reversed_cnt.shape[0]-1)
                    new_counter = reversed_cnt[frames_pass]
                    print('couting rewinded by %i' %(global_counter-new_counter))
                    global_counter = new_counter
                    # stop counting, move to cooldown
                    cur_state = state.COOLDOWN
                    # init cooldown counter
                    cooldown_in_time = in_frame_num/30
        if (cur_state == state.COOLDOWN):
            cooldown_out_time = in_frame_num/30
            if ((cooldown_out_time-cooldown_in_time)>4):
                global_counter = 0
                cur_state = state.NO_REP


def draw_str(dst, pos, s, color, scale):

    x = pos[0]
    y = pos[1]
    if (color[0]+color[1]+color[2]==255*3):
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness = 4, lineType=10)
    else:
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness = 4, lineType=10)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType=11)


def shared_dataset(data_xy, borrow=True):

    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32'), shared_y


def process_single_frame(frame):

    # convert to gray scal
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # downscale by 2
    gray_frame = gray_frame[::2,::2]
    return gray_frame



if __name__ == '__main__':

    rng = numpy.random.RandomState(23455)

    ######################## build start ########################

    # create an empty shared variables to be filled later
    data_x = numpy.zeros([1,20*50*50])
    data_y = numpy.zeros(20)
    train_set = (data_x, data_y)
    test_set_x, test_set_y, shared_test_set_y  = shared_dataset(train_set)

    print('building ... ')
    batch_size = 1

    D_model = RepetitionCountingNet('./weights.save')
    with open('model_save', 'wb') as f:
        torch.save(D_model, f)


    ######################## build done ###########################


    countMap = numpy.zeros([num_of_vids,10])
    for vidNum in range(1,num_of_vids+1):

        # input video
        cap = cv2.VideoCapture('../../data/mydata/YTIO_'+str(vidNum)+'.mov')
        # output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('../out/YTIO_out'+str(vidNum)+'.mov', fourcc, frame_rate, (640, 480))

        global_counter = 0
        winner_stride = 0
        cur_state = state.NO_REP;
        in_frame_num = -1
        actions_counter = 0

        ret, frame = cap.read()
        frame = scipy.misc.imresize(frame, size=(480,640),interp='bilinear')
        proFrame = process_single_frame(frame)

        # init detectors
        stA_det = RepDetector(D_model, proFrame, detector_strides[0])
        stB_det = RepDetector(D_model, proFrame, detector_strides[1])
        stC_det = RepDetector(D_model, proFrame, detector_strides[2])

        while True:

            in_frame_num += 1

            ret, frame = cap.read()
            if (ret == 0):
                print ('vid'+str(vidNum)+ ' done')
                break

            frame = scipy.misc.imresize(frame, size=(480,640),interp='bilinear')
            proFrame = process_single_frame(frame)

            # handle stride A
            if (in_frame_num % stA_det.stride_number == 0):
                stA_det.count(proFrame)
            # handle stride B
            if (in_frame_num % stB_det.stride_number == 0):
                stB_det.count(proFrame)
            # handle stride C
            if (in_frame_num % stC_det.stride_number == 0):
                stC_det.count(proFrame)

            # display result on video
            blue_color = (130, 0, 0)
            green_color = (0, 130, 0)
            red_color = (0, 0, 130)
            orange_color = (0,140,255)

            out_time = in_frame_num/30
            if  ((cur_state == state.IN_REP) and (((out_time-in_time)<4) or (global_counter<5))):
                draw_str(frame, (20, 120), " new hypothesis (%d) " % global_counter, orange_color, 1.5)
            if ((cur_state == state.IN_REP) and ((out_time-in_time)>=4) and (global_counter>=5)):
                draw_str(frame, (20, 120), "action %d: counting... %d" % (actions_counter, global_counter), green_color, 2)
            if ((cur_state == state.COOLDOWN) and (global_counter>=5)):
                draw_str(frame, (20, 120), "action %d: done. final counting: %d" % (actions_counter, global_counter), blue_color, 2)
                countMap[vidNum-1,actions_counter-1] = global_counter

            video_writer.write(frame)
            cv2.namedWindow('threaded video', cv2.WINDOW_NORMAL)
            cv2.imshow('threaded video', frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        # if video and done and we are in the middle of counting, take it into account
        if (cur_state == state.IN_REP):
            countMap[vidNum-1,actions_counter-1] = global_counter
        # close all
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()


    # done counting all videos
    # produce stats
    print('done. count map:')
    print(countMap)
    gt_counts = numpy.array([10,7,27,6,5,24,7,26,7,14,5,12,6,29,10,10,9,14,13,10,12,5,5,5,5])
    xx = numpy.sum(countMap, axis=1)
    numpy.abs(gt_counts - xx)
    dif = numpy.abs(gt_counts - xx)
    print ('number of vids that count diff = 0 =  % i' %(numpy.sum(dif < 1)))
    print ('number of vids that count diff <= 1 =  % i' %(numpy.sum(dif < 2)))
    print ('number of vids that count diff <= 2 =  % i' %(numpy.sum(dif < 3)))
    print ('number of vids that count diff <= 3 =  % i' %(numpy.sum(dif < 4)))
    print('done')
