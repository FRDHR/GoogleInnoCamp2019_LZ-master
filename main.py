from __future__ import print_function, division
import keras
import matplotlib.pyplot as plt
import sys,os
import numpy as np
from PIL import Image

generator_path = 'GANS/models/dress.h5'

class Generator:
    def __init__(self):
        self.generator = keras.models.load_model(generator_path)
        self.generator.summary()

    def save_imgs(self):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # Write these images to files
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        data_name = generator_path.split('/')[-1].split('.')[0]
        fig_name = "dcgan_%s.png" % data_name
        fig.savefig(fig_name)
        plt.close()

        return gen_imgs
def resize(img_name):
        img = Image.open(img_name)
        img2= img.crop((90,60,560,425))
        img2.save(img_name)


if __name__ == '__main__':
    generator = Generator()
    res=generator.save_imgs()
    resize("dcgan_dress.png")
    os.system("python ./SF/neural_style_transfer.py dcgan_dress.png ./SF/pics/1.jpg ./SF/pics/result")