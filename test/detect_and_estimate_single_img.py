import sys
sys.path.append(".")
import cv2
import numpy as np
from detector.detector import Detector
from detector.estimator import Estimator

from config import MODEL_WEIGHT_SAVE_DIR, TEST_INPUT_IMG_DIR, TEST_OUTPUT_IMG_DIR
from utils import draw_axis


def main(image_file):
    # read image file
    input_img_full_path = TEST_INPUT_IMG_DIR + '/' + image_file
    output_img_full_path = TEST_OUTPUT_IMG_DIR + '/' + image_file
    image = cv2.imread(input_img_full_path)

    # detection
    [h, w] = image.shape[:2]
    image_size = min(h, w)

    detector = Detector(weight_dir=MODEL_WEIGHT_SAVE_DIR, mode=2, min_face_size=image_size/10)
    bbox, bboxes = detector.predict(image)

    labels = bboxes[:, 4]
    bboxes = bboxes[np.argsort(labels)[-1:], :]     # TODO: test only


    print('bboxes-shape---:', bboxes.shape)
    num_boxes = bboxes.shape[0]

    # crop head pose estimate images
    square_boxes = detector.convert_to_square(bboxes)
    square_boxes[:, 0:4] = np.round(square_boxes[:, 0:4])

    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = detector.pad(square_boxes, w, h)

    cropped_ims = np.zeros((num_boxes, 64, 64, 3), dtype=np.float32)
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = image[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        cropped_ims[i, :, :, :] = cv2.resize(tmp, (64, 64)) / 255


    # estimate pose
    estimator = Estimator(model_path="../model_weight/pose_estimate_model.h5")
    _, poses = estimator.estimate(cropped_ims)

    # draw b-boxes and head axises in image
    for i in range(num_boxes):
        # print('bbox score--:',bbox[4])
        # cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        bbox_ = bboxes[i]
        pose_ = poses[i]
        cv2.rectangle(image, (int(bbox_[0]), int(bbox_[1])), (int(bbox_[2]), int(bbox_[3])), (0, 0, 255))
        tdy = round((bbox_[1]+bbox_[3])/2)
        tdx = round((bbox_[0]+bbox_[2])/2)
        image = draw_axis(image, yaw=pose_[0], pitch=pose_[1], roll=pose_[2], tdx=tdx, tdy=tdy)

    cv2.imwrite(output_img_full_path, image)
    # cv2.imshow('yy', image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("ERROR:%s Input img name with .jpg \r\n" % (sys.argv[0]))
    # else:
    #     main(sys.argv[1]
    main('07.png')
