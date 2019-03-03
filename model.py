import os
from scipy.misc import imsave
import numpy as np
import imageio
from tqdm import tqdm
import time

class Model(object):

    def update_params(self, input_tensor):
        '''Update parameters of the network

        Args:
            input_tensor: a batch of flattened images

        Returns:
            Current loss value
        '''
        raise NotImplementedError()

    def generate_and_save_compact_images(self, numsamples, seqLen, inputNoiseList, inputCleanOriList, directory, epoch, j, phase):
        imgs_folder = os.path.join(directory, 'imgEpoch%dIter%d_%s') % (epoch, j, phase)
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)

        noiseList = {i1: d1 for i1, d1 in zip(self.inputNoiseList, inputNoiseList)}
        denoisedImgList = self.sess.run(self.denoised_imgList, noiseList)

        with imageio.get_writer(os.path.join(imgs_folder, 'video.gif'), mode='I') as writer:
            for i in range(seqLen):    
                denoisedImg     = np.add(np.multiply(denoisedImgList[i], 128.), 128.)
                denoisedImg.reshape(numsamples, 128, 128, 3)
                inputNoise      = np.add(np.multiply(inputNoiseList[i], 128.), 128.)
                inputCleanOri   = inputCleanOriList[i]
                num = min(numsamples, 8)
                depthImg = self.toThreeC(inputNoise[0:num,:,:,3])
                normalImg = inputNoise[0:num,:,:,4:7]
                compact_img     = self.compact_batch_img([inputNoise[0:num,:,:,0:3],depthImg,normalImg,denoisedImg[0:num],inputCleanOri[0:num]], numsamples)
                compact_img = compact_img.astype(np.uint8)
                imsave(os.path.join(imgs_folder, '%d.png') % (i), compact_img)
                writer.append_data(compact_img)

    def generate_and_save_long_videos(self, numsamples, seqLen, inputNoiseList, inputCleanOriList, directory, epoch, j, phase):
        imgs_folder = os.path.join(directory, 'imgEpoch%dIter%d_%s') % (epoch, j, phase)
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)

        noiseList = {i1: d1 for i1, d1 in zip(self.inputNoiseList, inputNoiseList)}
        start_time = time.time()
        denoisedImgList = self.sess.run(self.denoised_imgList, noiseList)
        timeSpend = time.time() - start_time
        print("--- Total %s seconds ---" % (timeSpend))
        print("--- Average %s seconds ---" % (timeSpend/seqLen))
        with imageio.get_writer(os.path.join(imgs_folder, 'video.gif'), mode='I') as writer:
            for i in tqdm(range(seqLen)):    
                inputNoise      = np.add(np.multiply(inputNoiseList[i], 128.), 128.)
                inputCleanOri   = inputCleanOriList[i]
                denoisedImg     = np.add(np.multiply(denoisedImgList[i], 128.), 128.)
                denoisedImg.reshape(numsamples, inputNoise.shape[1], inputNoise.shape[2], 3)
                compact_img     = np.hstack([inputNoise[0,:,:,0:3],denoisedImg[0],inputCleanOri[0]])
                compact_img = compact_img.astype(np.uint8)
                writer.append_data(compact_img)

    def toThreeC(self, a):
        return np.stack([a,a,a], axis=3)

    def compact_batch_img(self, input_nparyList, batchSize):
        rowList = []
        for row in input_nparyList:
            compact_row = np.hstack([row[i] for i in range(min(8, batchSize))])
            rowList.append(compact_row)

        compact_img = np.vstack(rowList)

        return compact_img
