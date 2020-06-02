import cv2
import os



def main():
    # 这里放上自己所需要合成的图片
    image_dir = "/Users/leon/Downloads/DrivFace/drivface_outdir"
    img_list = os.listdir(image_dir)
    img_list.sort()

    fps = 12
    size = (640, 480)
    videowriter = cv2.VideoWriter("../test_output_img/test.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    path = image_dir
    for i in img_list:
        img = cv2.imread(path + '/' + i)
        videowriter.write(img)
    videowriter.release()


if __name__ == '__main__':
    main()