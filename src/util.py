# Author: Yan Zhang  
# Email: zhangyan.cse (@) gmail.com

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp2d

#constants
chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

# Read the HiC interaction matrix, which the format is sparse format
def readMatrix(filename, total_length):
    infile = open(filename).readlines()
    print("reading " + str(len(infile)))
    HiC = np.zeros((total_length,total_length)).astype(np.int16)
    percentage_finish = 0
    for i in range(0, len(infile)):
        if (i % (len(infile) / 10)== 0):
            print 'finish ', percentage_finish, '%'
            percentage_finish += 10
        nums = infile[i].split('\t')
        x = int(nums[0])
        y = int(nums[1])
        val = int(float(nums[2]))
        HiC[x][y] = val
        HiC[y][x] = val
    return HiC

def genSubRegions(HiCsample, result, index, matrix_name = '', SubRegion_size = 40, overlap = 15, max_distance = 100):
    step = SubRegion_size - overlap
    total_loci = HiCsample.shape[0]
    index.append((-1, total_loci))
    for i in range(0, total_loci, step):
        for j in range(0, total_loci, step):
            if (abs(i - j) > max_distance or i + SubRegion_size >= total_loci or j + SubRegion_size >= total_loci):
                continue
            subImage = HiCsample[i : i + SubRegion_size, j : j + SubRegion_size]
            result.append([subImage,])
            index.append((i, j))
    index.append((-1, -1))



def genInterpolatedHiC(inputMatrix, kind='cubic'):
    original_length = inputMatrix.shape[0]
    step = 40 / original_length
    x = np.arange(0, 40, step)
    y = np.arange(0, 40, step)
    xx, yy = np.meshgrid(x, y)
    z = inputMatrix
    f = interp2d(x, y, z, kind=kind)
    xnew = np.arange(0, 40, 1)
    ynew = np.arange(0, 40, 1)
    znew = f(xnew, ynew)
    return znew

def genLowResSamples(inputMatrix, scale):
    original_length = inputMatrix.shape[0]
    if (inputMatrix.shape[0] % scale != 0):
        print("size not fit")
        return
    new_length = original_length/scale
    result = np.zeros((new_length, new_length)).astype(np.float16)
    for i in range(0, original_length, scale):
        for j in range(0, original_length,scale):
            submatrix = inputMatrix[i:i+scale, j:j+scale]
            x = i/scale
            y = j/scale
            result[x][y] = np.mean(submatrix)
    return result

def genScaledLowRes(inputMatrix, scale):
    original_length = inputMatrix.shape[0]
    step = scale
    new_length = scale * original_length
    result = np.zeros((new_length, new_length))
    for i in range(0, original_length):
        for j in range(0, original_length):
            result[i*step:i*step+step, j*step:j*step+step] = inputMatrix[i][j]
    return result












def main():
    cells =['GM12878_combined','GM12878','GM12878_replicate',  'HMEC',  'HUVEC',  'IMR90',  'K562',  'KBM7',  'NHEK']
    
    #cells =[ 'HUVEC']
    #cells =['K562', 'IMR90']
    #cells =['NHEK', 'KBM7',]
    cells =[ 'GM12878_primary']
    cells =['K562', 'KBM7']
    cells = ['CH12-LX']
    step = 25
    subImage_size = 40

    chrN_start = 1
    chrN_end = 5
    for cell in cells:
        result = []
        index = []
        for chrN in range(chrN_start,chrN_end+1, 1):
            matrix_name = '/home/zhangyan/standard_hic_sample/samples/primary_OriginalHiC_10k_e30_' +cell+ '_chr' + str(chrN) + '.npy'
            if os.path.exists(matrix_name):
                print 'loading ', matrix_name
                HiCsample = np.load(matrix_name)
            else:
                print matrix_name, 'not exist, creating'
                HiCfile = '/home/zhangyan/normHiC10k/'+cell+'chr'+str(chrN)+'_norm_10k_maqe30.hic'
                print HiCfile           
                HiCsample = readMatrix(HiCfile, (chrs_length[chrN-1]/10000 + 1))
                np.save(matrix_name, HiCsample)

            path = '/home/zhangyan/standard_hic_sample/original_10k_30/' + cell
            if not os.path.exists(path):
                os.makedirs(path)
            total_loci = chrs_length[chrN-1]/10000
            for i in range(0, total_loci, step):
                for j in range(0, total_loci, step):
                    if (abs(i-j) > 101 or i + subImage_size >= total_loci or j + subImage_size >= total_loci):
                        continue
                    subImage = HiCsample[i:i+subImage_size, j:j+subImage_size]

                    result.append([subImage,])
                    index.append((cell, chrN, i, j))
        result = np.array(result)
        print result.shape
        result = result.astype(np.int16)
        np.save('/home/zhangyan/standard_hic_sample/samples/original10k/'+cell+'_original_chr'+str(chrN_start)+'_' + str(chrN_end), result)
        index = np.array(index)
        np.save('/home/zhangyan/standard_hic_sample/samples/original10k/'+cell+'_original_index_chr'+str(chrN_start)+'_' + str(chrN_end), index)




    #heatmap = plt.imshow(HiCsample[i], cmap=plt.cm.Reds, interpolation='nearest', origin='lower', vmax = 25)
    #plt.show()


  
if __name__ == "__main__":
    main()