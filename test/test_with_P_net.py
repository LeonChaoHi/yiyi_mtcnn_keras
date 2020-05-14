
import sys
sys.path.append(".")
import cv2
import numpy as np
from detector.detector import Detector

from config import MODEL_WEIGHT_SAVE_DIR, TEST_INPUT_IMG_DIR, TEST_OUTPUT_IMG_DIR


def main(image_file):
    # read image file
    input_img_full_path = TEST_INPUT_IMG_DIR + '/' + image_file
    output_img_full_path = TEST_OUTPUT_IMG_DIR + '/' + image_file
    image = cv2.imread(input_img_full_path)

    # detection
    image_size = min(image.shape[0], image.shape[1])
    detector = Detector(weight_dir=MODEL_WEIGHT_SAVE_DIR, mode=2, min_face_size=image_size/10)
    bbox, bboxes = detector.predict(image)
    labels = bboxes[:, 4]

    w = np.array(bboxes[:, 2] - bboxes[:, 0])
    h = np.array(bboxes[:, 3] - bboxes[:, 1])

    print('bboxes-shape---:', bboxes.shape)

    bboxes_ranked = bboxes[np.argsort(labels)[:], :]
    # bboxes_ranked = bboxes

    for bbox_ in bboxes_ranked:
        # print('bbox score--:',bbox[4])
        # cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(image, (int(bbox_[0]), int(bbox_[1])), (int(bbox_[2]), int(bbox_[3])), (0, 0, 255))


    cv2.imwrite(output_img_full_path, image)
    # cv2.imshow('yy', image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("ERROR:%s Input img name with .jpg \r\n" % (sys.argv[0]))
    # else:
    #     main(sys.argv[1]
    main('01.jpg')
