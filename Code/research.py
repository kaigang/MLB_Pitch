import moviepy.editor as mp
import os
import cv2 as cv
import imageio
import numpy as np

def resize_videos(files,height = 720):

    for file in files:
        print('resizing %s to *_resized.mp4' % file)
        clip = mp.VideoFileClip(file)
        clip_resized = clip.resize(
            height=height)
        save_path = os.path.join(os.path.dirname(file),str(height),os.path.basename(file))
        clip_resized.write_videofile(save_path)

def get_flow(file):
    print('processing %s' % file)
    cap = cv.VideoCapture(file)
    vid = imageio.get_reader(file, 'ffmpeg')

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=3,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    first_frame = vid.get_data(index=1)
    mask = np.zeros_like(first_frame)
    for i in range(int(vid.get_length()/2)):
        cur = vid.get_data(index=2*i)
        next = vid.get_data(index=2*i+1)
        cur_gray = cv.cvtColor(cur,cv.COLOR_BGR2GRAY)
        next_gray = cv.cvtColor(next,cv.COLOR_BGR2GRAY)
        p0 = cv.goodFeaturesToTrack(cur_gray, mask = None, **feature_params)
        p1, st, err = cv.calcOpticalFlowPyrLK(cur_gray, next_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), [255,255,255], 2)
            first_frame = cv.circle(first_frame,(a,b),10,color[i].tolist(),-1)
        img = cv.add(first_frame,mask)
    cv.imwrite(file[0:-4] + '_mask.png', mask)
    cap.release()
    vid.close()
    return mask

