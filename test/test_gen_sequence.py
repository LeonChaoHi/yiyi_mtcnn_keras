import sys
sys.path.append(".")
import cv2
import numpy as np
from detector.detector import Detector
from detector.estimator import Estimator
from matplotlib import pyplot as plt
import scipy.signal as signal
import os
import time

from config import MODEL_WEIGHT_SAVE_DIR, TEST_INPUT_IMG_DIR, TEST_OUTPUT_IMG_DIR
from utils import draw_axis


def main(image_file):
    image_dir = "/Users/leon/Downloads/DrivFace/drivface3"
    img_list = os.listdir(image_dir)
    img_list.sort()
    output_dir = "/Users/leon/Downloads/DrivFace/"

    # init detector and estimator
    detector = Detector(weight_dir=MODEL_WEIGHT_SAVE_DIR, mode=2, min_face_size=48)
    estimator = Estimator(model_path="../model_weight/pose_estimate_model.h5")

    # init variables
    w = 640
    h = 480
    img_count = 0
    fail_list = []
    start = time.time()

    # init output list
    output = []

    for img_name in img_list:
        print(img_name)
        img_count += 1

        # read image file
        input_img_full_path = os.path.join(image_dir, img_name)
        image = cv2.imread(input_img_full_path)

        # detection
        bbox, bboxes = detector.predict(image)
        if bboxes is None:
            fail_list.append(img_count)
            continue

        labels = bboxes[:, 4]
        width = bboxes[:, 2] - bboxes[:, 0]
        bboxes = bboxes[width > 50]
        bboxes = bboxes[np.argsort(labels)[-1:], :]     # TODO: test only

        # crop head pose estimate images
        square_boxes = detector.convert_to_square(bboxes)
        square_boxes[:, 0:4] = np.round(square_boxes[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = detector.pad(square_boxes, w, h)

        cropped_ims = np.zeros((1, 64, 64, 3), dtype=np.float32)
        for i in range(1):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = image[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = cv2.resize(tmp, (64, 64)) / 255

        # estimate pose
        _, poses = estimator.estimate(cropped_ims)

        output.append(poses)

        # draw b-boxes and head axises in image
        output_img_full_path = "/Users/leon/Downloads/DrivFace/drivface_outdir/" + img_name
        for i in range(1):
            # print('bbox score--:',bbox[4])
            bbox_ = bboxes[i]
            pose_ = poses[i]
            # cv2.rectangle(image, (int(bbox_[0]), int(bbox_[1])), (int(bbox_[2]), int(bbox_[3])), (0, 0, 255))
            tdy = round((bbox_[1] + bbox_[3]) / 2)
            tdx = round((bbox_[0] + bbox_[2]) / 2)
            image = draw_axis(image, yaw=pose_[0], pitch=pose_[1], roll=pose_[2], tdx=tdx, tdy=tdy)

        cv2.imwrite(output_img_full_path, image)

    time_span = time.time() - start
    print('Time spent: ', time_span)

    # show and save result sequence
    output = np.array(output).squeeze(axis=1)
    yaw = output[:, 0]
    pitch = output[:, 1]
    roll = output[:, 2]
    # np.savez(os.path.join(output_dir, '03_pose_sequence'), yaw=yaw, pitch=pitch, roll=roll)

    # plot training curve
    axisx = range(1, img_count + 1)
    plt.plot(axisx, signal.medfilt(yaw, 3), color='b', label='yaw')
    plt.plot(axisx, signal.medfilt(pitch, 3), color='r', label='pitch')
    plt.plot(axisx, signal.medfilt(roll, 3), color='y', label='roll')
    plt.xlabel('frame')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("ERROR:%s Input img name with .jpg \r\n" % (sys.argv[0]))
    # else:
    #     main(sys.argv[1]
    main('demo_01.jpg')
