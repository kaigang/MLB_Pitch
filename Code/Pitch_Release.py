
import moviepy.editor as mp
import os
import cv2 as cv
import imageio
import numpy as np


class pitch_analysis(object):
    def __init__(self,vid_file):
        # takes video file name as input, returns the frame in which the player releases the ball from his hand

        self.vid_file = vid_file
        self.mask = None
        self.release_location = None
        self.release_index = None
        self.release_frame = None
    def _run(self):
        # a warpper to utlize all functions
        mask = self._calculate_opticalflow()
        self.release_location = self._calculate_lines(mask)
        temp = self._calculate_opticalflow(release_location=self.release_location)

    def _calculate_opticalflow(self, release_location = None):
        # takes video file name as input, returns the aggregated optical flow (between two frames)
        # also returns the frame if optional input release location is given.
        if release_location:
            print('calculating ball release frame')
        else:
            print('calculate optical flow from file: %s' % self.vid_file)
        try:
            cap = cv.VideoCapture(self.vid_file)
            vid = imageio.get_reader(self.vid_file, 'ffmpeg')
        except:
            return 'Video file can not be loaded'
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
        color = np.random.randint(0, 255, (1, 3))

        first_frame = vid.get_data(index=1)
        mask = np.zeros_like(first_frame)
        dist = 1000
        for j in range(int(vid.get_length() / 2)):
            cur = vid.get_data(index=2 * j)
            next = vid.get_data(index=2 * j + 1)
            cur_gray = cv.cvtColor(cur, cv.COLOR_BGR2GRAY)
            next_gray = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
            p0 = cv.goodFeaturesToTrack(cur_gray, mask=None, **feature_params)
            p1, st, err = cv.calcOpticalFlowPyrLK(cur_gray, next_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (a, b), (c, d), [255, 255, 255], 2)
                if self.release_location:
                    temp = np.sqrt(np.power(self.release_location[0]-a,2)+np.power(self.release_location[1]-b,2))
                    if temp<dist:
                        self.release_index = 2*j+1
                        dist = temp
                        self.release_frame = cv.circle(cur,(self.release_location[0],self.release_location[1]),5,color[0].tolist(),-1)
                        self.release_frame = cv.line(self.release_frame,tuple(self.release_location[0:2]) , tuple(self.release_location[2:4]) , (255, 0, 0), 5)
        cap.release()
        vid.close()
        return mask

    def _calculate_lines(self, mask):
        # takes optical flow mask as input and returns the ball trajectory (assuming it's a straight line )
        # it uses morphological transformations to reduce the noise

        img_bw = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv.dilate(img_bw, kernel).astype('uint8')
        # remove the largest connected components (the pitcher's movements caused)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(opening, connectivity=4)
        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        opening[output == largest_label] = 0

        erosion = cv.erode(opening, kernel, iterations=1)
        low_threshold = 50
        high_threshold = 150
        edges = cv.Canny(erosion, low_threshold, high_threshold)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 30  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 60  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments

        # Hough line detection
        lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                               min_line_length, max_line_gap)

        # in the results lines from the detection, find the starting and the ending points which will be used for
        # finding the release frame

        x_start = img_bw.shape[1]
        y_start = 0
        x_end = 0
        y_end = img_bw.shape[0]
        for line in lines:
            for x1, y1, x2, y2 in line:
                #         cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),1)
                x_start = min(x1, x_start)
                y_start = max(y1, y_start)
                x_end = max(x_end, x2)
                y_end = min(y_end, y2)
        return [x_end,y_end,x_start,y_start]
