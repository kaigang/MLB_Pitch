import os

TEST1 = '/home/kaigang/Desktop/MLB/Data/test1.mp4'
TEST2 = '/home/kaigang/Desktop/MLB/Data/test2.mp4'
TEST3 = '/home/kaigang/Desktop/MLB/Data/test3.mp4'
TEST4 = '/home/kaigang/Desktop/MLB/Data/test4.mp4'
TEST5 = '/home/kaigang/Desktop/MLB/Data/test5.mp4'

TEST_LINKS = [TEST1,TEST2,TEST3,TEST4,TEST5]

DATA_FILE = '/home/kaigang/Desktop/MLB/Data/'

TEST_RESIZED = lambda x: [DATA_FILE + str(x) +'/' + os.path.basename(file) for file in TEST_LINKS]


MASKS_FILES = lambda x: [DATA_FILE + str(x) +'/' + os.path.basename(file)[0:-4] + '_mask.png' for file in TEST_LINKS]


# TEST_STABLIZED = [file[0:-4] + '_stable.mp4' for file in TEST_LINKS]