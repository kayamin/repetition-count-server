import numpy
import scipy
from scipy.ndimage import filters


def get_boundingbox(frame_set):
    import pdb
    pdb.set_trace()
    fstd = numpy.std(frame_set, axis=0)
    framesstd = numpy.mean(fstd)
    th = framesstd
    ones = numpy.ones(10)
    big_var = (fstd > th)

    if (framesstd == 0):  # no bb, take full frame
        frameROIRes = numpy.zeros([20, 50, 50])
        for i in range(20):
            frameROIRes[i, :, :] = scipy.misc.imresize(frame_set[i, :, :], size=(50, 50), interp='bilinear')

        frameROIRes = numpy.reshape(frameROIRes,
                                    (1, frameROIRes.shape[0] * frameROIRes.shape[1] * frameROIRes.shape[2]))
        frameROIRes = frameROIRes.astype(numpy.float32)

        return (frameROIRes, framesstd)

    big_var = big_var.astype(numpy.float32)
    big_var = filters.convolve1d(big_var, ones, axis=0)
    big_var = filters.convolve1d(big_var, ones, axis=1)

    th2 = 80
    i, j = numpy.nonzero(big_var > th2)

    if (i.size > 0):
        si = numpy.sort(i)
        sj = numpy.sort(j)

        ll = si.shape[0]
        th1 = round(ll * 0.02)
        th2 = numpy.floor(ll * 0.98).astype('int')
        y1 = si[th1]
        y2 = si[th2]
        x1 = sj[th1]
        x2 = sj[th2]

        # cut image ROI
        if (((x2 - x1) > 0) and ((y2 - y1) > 0)):
            framesRoi = frame_set[:, y1:y2, x1:x2]
        else:
            framesRoi = frame_set[:, :, :]
    else:
        framesRoi = frame_set[:, :, :]

    # debug - show ROI
    # cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
    # bla= scipy.misc.imresize(framesRoi[19,:,:], size=(200,200),interp='bilinear')
    # cv2.imshow('ROI', bla)

    # resize to 50x50
    frameROIRes = numpy.zeros([20, 50, 50])
    for i in range(20):
        frameROIRes[i, :, :] = scipy.misc.imresize(framesRoi[i, :, :], size=(50, 50), interp='bilinear')

    import pdb
    pdb.set_trace()

    riofstd = numpy.std(frameROIRes, axis=0)
    cur_std = numpy.mean(riofstd)

    # return 4d ndarray
    frameROIRes = frameROIRes.reshape(-1, 20, 50, 50).astype(numpy.float32)

    return frameROIRes, cur_std


if __name__=='__main__':

    # 20 frame の中で 20x20 の正方形が移動しているとする真ん中で左端から右端まで0, 7 移動していく
    frames = numpy.zeros([20, 120, 160]).astype('uint8')
    for i in range(20):
        # frames[i, 100:140, 14*i:14*i+40] = numpy.ones([40, 40]).astype('uint8') * 255
        frames[i, 50:70, 7*i:7*i+20] = numpy.ones([20, 20]).astype('uint8') * 255

    # frames = frames.astype('float') / 255
    roi, cur_std = get_boundingbox(frames)
    import pdb
    pdb.set_trace()