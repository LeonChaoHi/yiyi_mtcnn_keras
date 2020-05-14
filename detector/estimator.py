from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Estimator:
    def __init__(self, model_path='../weights_file/model_parameters (2).h5', mode='API'):
        assert mode in ['API', 'test']
        self.mode = mode
        self.model = load_model(model_path)

    def estimate(self, input_img=None, n=1):

        if self.mode == 'test':     # test mode
            npzdata = np.load('../gen_data/hpdb_data2.npz')
            train_x, train_y, test_x, test_y = npzdata['train_x'], npzdata['train_y'], npzdata['test_x'], npzdata['test_y']

            x = train_x[(n,), :, :, :] / 255   # normalize to [0,1], shape: (1, 64, 64, 1)

            print('Sample NO.: ', n)
            print('Ground Truth: ', train_y[n, :])    # give ground truth

        else:
            if not (type(input_img) is np.ndarray and input_img.ndim == 4):
                raise TypeError('Please input numpy array format image with shape as [N, W, H, 3]')
            img_num = input_img.shape[0]
            x = np.zeros([img_num, 64, 64])
            # Turn gray, Gamma correction and resize
            for i in range(img_num):
                if input_img.shape[1:3] != (64, 64):
                    img_resized = cv2.resize(input_img[i], (64, 64))        # resize    # TODO: Change to 4 dim
                else:
                    img_resized = input_img[i]
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)    # turn gray
                x[i] = np.power(img_gray / np.max(img_gray), 1 / 1.5)        # gamma --> [0, 1]

            x = x[:, :, :, np.newaxis]               # shape: (N, 64, 64, 1)

        # predict with model
        y = self.model.predict(x)

        # revert normalization
        pred = y * np.array([120, 150, 100]) - np.array([60, 75, 50])

        img = x.squeeze() * 255

        return img, pred


if __name__ == '__main__':
    estimator = Estimator(mode='test')
    img, pred = estimator.estimate(n=1)

    print('Pred: ', pred)
    plt.imshow(img)
    plt.show()

