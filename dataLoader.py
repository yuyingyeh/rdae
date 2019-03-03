import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import scipy.signal as ssig
from itertools import product
from tqdm import tqdm

class BatchLoader(object):
    def __init__(self, dataRoot, inpChannel = 3, batchSize = 32, seqLen = 7, scaleSize = (425, 800), cropSize=(128, 128),
            isRandom=True, phase='TRAIN', rseed = None, loadDataOriginal=True):
        
        self.dataRoot = dataRoot
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.loadDataOriginal = loadDataOriginal
        self.inpChannel = inpChannel

        self.scaleH = scaleSize[0]
        self.scaleW = scaleSize[1]
        self.cropH = cropSize[0]
        self.cropW = cropSize[1]
        assert(self.cropH <= self.scaleH)
        assert(self.cropW <= self.scaleW)
        self.phase = phase.upper()
        
        # Get Image List:
        if (phase == 'train') or (phase == 'test'):
            self.imageListNoise = self.getOrCreateSplits(dataRoot, seqLen, phase)
            self.count = len(self.imageListNoise)
            if rseed is not None:
                random.seed(rseed)
    
            self.imageListClean = [ [img[:img.find('X')]+'Y.png' for img in seq] for seq in self.imageListNoise]
            if inpChannel > 3:
                self.imageListDepth = [ [img[:img.find('X')]+'X_depth.png' for img in seq] for seq in self.imageListNoise]
                self.imageListNormalX = [ [img[:img.find('X')]+'X_shading_x.png' for img in seq] for seq in self.imageListNoise]
                self.imageListNormalY = [ [img[:img.find('X')]+'X_shading_y.png' for img in seq] for seq in self.imageListNoise]
                self.imageListNormalZ = [ [img[:img.find('X')]+'X_shading_z.png' for img in seq] for seq in self.imageListNoise]

            self.perm = list(range(self.count) )
            
            if isRandom:
                random.shuffle(self.perm)

            
            
        if phase == 'predict':
            self.imageListNoise = self.getNoiseLongList(dataRoot, seqLen)
            print(self.imageListNoise)
            self.imageListClean = [ [img[:img.find('X')]+'Y.png' for img in seq] for seq in self.imageListNoise]
            if inpChannel > 3:
                self.imageListDepth = [ [img[:img.find('X')]+'X_depth.png' for img in seq] for seq in self.imageListNoise]
                self.imageListNormalX = [ [img[:img.find('X')]+'X_shading_x.png' for img in seq] for seq in self.imageListNoise]
                self.imageListNormalY = [ [img[:img.find('X')]+'X_shading_y.png' for img in seq] for seq in self.imageListNoise]
                self.imageListNormalZ = [ [img[:img.find('X')]+'X_shading_z.png' for img in seq] for seq in self.imageListNoise]
            
            self.count = len(self.imageListNoise)
            self.perm = list(range(self.count) )

        self.inputBatchNoise = np.zeros( (batchSize, self.seqLen, self.cropH, self.cropW, inpChannel), dtype=np.float32 )
        self.inputBatchClean = np.zeros( (batchSize, self.seqLen, self.cropH, self.cropW, 3), dtype=np.float32 )
        
        self.inputBatchNoiseOriginal = np.zeros( (batchSize, self.seqLen, self.cropH, self.cropW, 3), dtype=np.float32 )
        self.inputBatchCleanOriginal = np.zeros( (batchSize, self.seqLen, self.cropH, self.cropW, 3), dtype=np.float32 )

        self.cur = 0      

    def getOrCreateSplits(self, dataRoot, seqLen, phase):
        print(osp.join(dataRoot, 'splittrain.npy'))
        if not osp.isfile(osp.join(dataRoot, 'splittrain.npy')) :
            # create list of image sequences, data aug with noise sampling and forward/backward
            split = 0.9
            sceneIdx = list(range(41+14))
            sceneIdx.remove(31)
            sceneIdx.remove(33)
            sceneIdx = np.asarray(sceneIdx)
            random.shuffle(sceneIdx)
            trainIdx = sceneIdx[0:int(round(len(sceneIdx)*split))]
            testIdx = sceneIdx[int(round(len(sceneIdx))*split):-1]

            for t, Idx in enumerate([trainIdx, testIdx]):
                if t == 0:
                    print('Creating training list...')
                if t == 1:
                    print('Creating testing list...')
                seqList = []
                for seqId in Idx:
                    if seqId < 41:
                        print('Processing ', seqId, 'th pavilion sequence...')
                    else:
                        print('Processing ', str(seqId % 41), 'th bathroom sequence...')
                    for comb in tqdm(product(range(4), repeat=seqLen)):
                        comb = list(comb) # convert tuple to list, choice of each time step
                        seq1 = []
                        seq2 = []
                        for i, choice in enumerate(comb):
                            if seqId < 41:
                                seq1.append(osp.join(dataRoot, 'pavilion_{0}_{1}_X_{2}.png'.format(seqId, i, choice)))
                                seq2.append(osp.join(dataRoot, 'pavilion_{0}_{1}_X_{2}.png'.format(seqId, seqLen-1-i, choice)))
                            else:
                                seq1.append(osp.join(dataRoot, 'bathroom_{0}_{1}_X_{2}.png'.format(seqId%41, i, choice)))
                                seq2.append(osp.join(dataRoot, 'bathroom_{0}_{1}_X_{2}.png'.format(seqId%41, seqLen-1-i, choice)))
                        seqList.append(seq1+seq2)

                random.shuffle(seqList)
                if t == 0:
                    np.save(osp.join(dataRoot, 'splittrain.npy'), seqList)
                if t == 1:
                    np.save(osp.join(dataRoot, 'spliteval.npy'), seqList)

            print(' Successfully Created New Split')
        
        if phase == 'train':
            seqList = np.load(osp.join(dataRoot, 'splittrain.npy'))
            print(' Loaded Training Split')
        if phase == 'test':
            seqList = np.load(osp.join(dataRoot, 'splitval.npy'))
            print(' Loaded Validation Split')
        print(' Training Samples: '+str(len(seqList)))
        return seqList


    def getNoiseLongList(self, dataRoot, seqLen, seqNum=1):
        Idx = [0]
        seqList = []
        for seqId in Idx:
            print('Processing ', seqId, 'th sequence...')
            for _ in range(seqNum):
                comb = [np.random.randint(10) for _ in range(seqLen)]
                print(comb)
                seq1 = []
                seq2 = []
                for i, choice in enumerate(comb):
                    seq1.append(osp.join(dataRoot, 'pavilion_{0}_{1}_X_{2}.png'.format(seqId, i, choice)))
                    seq2.append(osp.join(dataRoot, 'pavilion_{0}_{1}_X_{2}.png'.format(seqId, seqLen-1-i, choice)))
                seqList.append(seq1+seq2)
        return seqList

    def loadBatch(self):
        isNewEpoch = False
        for i in range(0, self.batchSize):
            if self.cur == self.count:
                self.cur = 0
                random.shuffle(self.perm)
                isNewEpoch = True
                if self.phase == 'test':
                    break   
            
            if self.inpChannel == 4:
                inputSeqNoiseList, inputSeqCleanList, inputSeqDepthList, \
                    inputSeqNoiseListOriginal, inputSeqCleanListOriginal, inputSeqDepthListOriginal = \
                    self.loadImageList([self.imageListNoise[self.perm[self.cur] ], \
                                    self.imageListClean[self.perm[self.cur] ],
                                    self.imageListDepth[self.perm[self.cur]]
                                    ])
            elif self.inpChannel == 7:
                inputSeqNoiseList, inputSeqCleanList, inputSeqDepthList, \
                    inputSeqNormalXList, inputSeqNormalYList, inputSeqNormalZList, \
                    inputSeqNoiseListOriginal, inputSeqCleanListOriginal, inputSeqDepthListOriginal, \
                    inputSeqNormalXListOriginal, inputSeqNormalYListOriginal, inputSeqNormalZListOriginal = \
                    self.loadImageList([self.imageListNoise[self.perm[self.cur] ], \
                                    self.imageListClean[self.perm[self.cur] ],
                                    self.imageListDepth[self.perm[self.cur]],
                                    self.imageListNormalX[self.perm[self.cur]],
                                    self.imageListNormalY[self.perm[self.cur]],
                                    self.imageListNormalZ[self.perm[self.cur]]
                                    ])
            inputSeqNoise, inputSeqNoiseOriginal = inputSeqNoiseList, inputSeqNoiseListOriginal
            inputSeqClean, inputSeqCleanOriginal = inputSeqCleanList, inputSeqCleanListOriginal
            inputSeqDepth, inputSeqDepthOriginal = inputSeqDepthList, inputSeqDepthListOriginal
            inputSeqNormalX, inputSeqNormalXOriginal = inputSeqNormalXList, inputSeqNormalXListOriginal
            inputSeqNormalY, inputSeqNormalYOriginal = inputSeqNormalYList, inputSeqNormalYListOriginal
            inputSeqNormalZ, inputSeqNormalZOriginal = inputSeqNormalZList, inputSeqNormalZListOriginal

            for j in range(self.seqLen):
                self.inputBatchNoise[i, j, :] = np.concatenate([inputSeqNoise[j], inputSeqDepth[j], inputSeqNormalX[j], inputSeqNormalY[j], inputSeqNormalZ[j]], axis=2)
                self.inputBatchClean[i, j, :] = inputSeqClean[j]
                self.inputBatchNoiseOriginal[i, j, :] = inputSeqNoiseOriginal[j]
                self.inputBatchCleanOriginal[i, j, :] = inputSeqCleanOriginal[j]
        
            self.cur += 1

        inputBatchNoiseList = []
        inputBatchNoiseOriginalList = []
        inputBatchCleanList = []
        inputBatchCleanOriginalList = []
        for k in range(self.seqLen):
            inputBatchNoiseList.append(self.inputBatchNoise[:, k, :, :, :])
            inputBatchNoiseOriginalList.append(self.inputBatchNoiseOriginal[:, k, :, :, :])
            inputBatchCleanList.append(self.inputBatchClean[:, k, :, :, :])
            inputBatchCleanOriginalList.append(self.inputBatchCleanOriginal[:, k, :, :, :])

        return inputBatchNoiseList, inputBatchNoiseOriginalList, inputBatchCleanList, inputBatchCleanOriginalList, isNewEpoch

    def loadImageList(self, noiseAndCleanSeq):
        noiseAndClean = []
        noiseAndCleanOri = []
        choice = np.random.randint((self.scaleH-self.cropH+1)*(self.scaleW-self.cropW+1))
        for seq in noiseAndCleanSeq: # first is noise, second is clean, third is depth, forth is normal
            im_list = []
            im_org_list = []
            for imName in seq:
                im = Image.open(imName)
                im = im.resize((self.scaleW, self.scaleH), Image.ANTIALIAS)
                im = np.asarray(im, dtype=np.float32)
                if len(im.shape)>2 :
                    im = im[:,:,:3] # remove alpha channel from image, if any
                else:
                    im = np.expand_dims(im, axis=2)
                # Randomly crop 128*128
                im = self.randomCrop(im, choice)
                
                im_org = im
                im = (im - 128.0)/128.0

                im_list.append(im)
                im_org_list.append(im_org)
            noiseAndClean.append(im_list)
            noiseAndCleanOri.append(im_org_list)

        inputNoiseSeq = noiseAndClean[0]
        inputNoiseSeqOri = noiseAndCleanOri[0]
        outputCleanSeq = noiseAndClean[1] 
        outputCleanSeqOri = noiseAndCleanOri[1]
        inputDepthSeq = noiseAndClean[2]
        inputDepthSeqOri = noiseAndCleanOri[2]
        inputNormalXSeq = noiseAndClean[3]
        inputNormalXSeqOri = noiseAndCleanOri[3]
        inputNormalYSeq = noiseAndClean[4]
        inputNormalYSeqOri = noiseAndCleanOri[4]
        inputNormalZSeq = noiseAndClean[5]
        inputNormalZSeqOri = noiseAndCleanOri[5]

        return inputNoiseSeq, outputCleanSeq, inputDepthSeq, inputNormalXSeq, inputNormalYSeq, inputNormalZSeq, \
            inputNoiseSeqOri, outputCleanSeqOri, inputDepthSeqOri, inputNormalXSeqOri, inputNormalYSeqOri, inputNormalZSeqOri

    def randomCrop(self, im, choice):
        pointH = choice // (self.scaleW-self.cropW+1)
        pointW = choice % (self.scaleW-self.cropW+1)
        im = im[pointH:pointH+self.cropH, pointW:pointW+self.cropW, :]
        return im
