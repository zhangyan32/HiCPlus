# Author: Yan Zhang  
# Email: zhangyan.cse@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import os
import util

resolution = 10000

def main():
    cells =['GM12878_combined','GM12878','GM12878_replicate',  'HMEC',  'HUVEC',  'IMR90',  'K562',  'KBM7',  'NHEK']

    chrN_start = 18
    chrN_end = 18

    cell = 'GM12878_replicate'
    file_list = []
    length_list = []
    for chrN in range(chrN_start,chrN_end+1, 1):
        HiCfile = '/home/zhangyan/normHiC10k/'+cell+'chr'+str(chrN)+'_norm_10k_maqe30.hic'
        file_list.append(HiCfile)
        length_list.append(util.chrs_length[chrN-1]/resolution) 
    genSamples(file_list, length_list, '/home/zhangyan/temptesttemptest', tag = str(chrN_start) + '_' + str(chrN_end)+ '_' + cell)
 

def genSamples(list_of_files, list_of_length, destination_folder, save_intermediate = True, load_intermediate = True, tag = ''):    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    index = []
    Exper_HiRes = []
    for i in range(0, len(list_of_files)):
        intermediate_file = destination_folder + '/' + list_of_files[i].replace('/', '_')
        if os.path.exists(intermediate_file + '.npy'):
            print 'loading intermediate file from ', intermediate_file
            HiCmatrix = np.load(intermediate_file + '.npy')
        else:   
            print(intermediate_file + ' does not exist, creating')      
            HiCmatrix = util.readMatrix(list_of_files[i], list_of_length[i])
            print(intermediate_file + ' is saving')
            np.save(intermediate_file, HiCmatrix)
        util.genSubRegions(HiCmatrix, Exper_HiRes, index)
    np.save(destination_folder + '/index' + tag, np.array(index))
    blurred_samples = []
    bicubic_samples = []
    bilinear_samples = []
    scaledLowRes_samples = []


    for i in range(0, len(Exper_HiRes)):
        Exper_HiRes_sample = np.minimum(100, Exper_HiRes[i][0])           
        blurred_sample = util.genLowResSamples(Exper_HiRes_sample, 4)
        bicubic_sample = util.genInterpolatedHiC(blurred_sample)
        bilinear_sample = util.genInterpolatedHiC(blurred_sample, kind='linear')
        scaled_lowRes = util.genScaledLowRes(blurred_sample, 4)
        blurred_samples.append([blurred_sample,])
        bicubic_samples.append([bicubic_sample,])
        bilinear_samples.append([bilinear_sample])
        scaledLowRes_samples.append([scaled_lowRes,])

    blurred_samples = np.array(blurred_samples).astype(np.float16)
    bicubic_samples = np.array(bicubic_samples).astype(np.float16)
    bilinear_samples = np.array(bilinear_samples).astype(np.float16)
    scaledLowRes_samples = np.array(scaledLowRes_samples).astype(np.float16)
    Exper_HiRes = np.minimum(100, np.array(Exper_HiRes).astype(np.float16))

    np.save(destination_folder + '/bilinear_x4_chr' + tag, bilinear_samples)
    np.save(destination_folder + '/blurred_x4_chr' + tag, blurred_samples)
    np.save(destination_folder + '/bicubic_x4_chr' + tag, bicubic_samples)
    np.save(destination_folder + '/scaled_blur_x4_chr' + tag, scaledLowRes_samples)
    np.save(destination_folder + '/experimentalHiCRes_chr' + tag, Exper_HiRes)


if __name__ == "__main__":
    main()










