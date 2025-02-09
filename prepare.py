# coding=utf-8
import os
import shutil
from config_training import config
from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial
import sys
sys.path.append('../preprocessing')
from preprocessing.step1 import step1_python
import warnings


def resample(imgs, spacing, new_spacing,order=2): # 对图像进行重采样，使得单个像素长宽高为1*1*1毫米
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')
def worldToVoxelCoord(worldCoord, origin, spacing): # 真实世界的坐标到立体像素坐标的转换
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):# 读取ITK图片，返回图片的numpy数组、坐标原点、单个像素的长宽高、是否翻转
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip

def process_mask(mask):# 
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask


def lumTrans(img):# 将图片像素值映射到0-256
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


def savenpy(id,annos,filelist,data_path,prep_folder):# 这是针对DSB数据的处理，因为我们没有DSB数据，所以并没有调用这个。
    resolution = np.array([1,1,1])
    name = filelist[id]
    label = annos[annos[:,0]==name]
    label = label[:,[3,1,2,4]].astype('float')

    im, m1, m2, spacing = step1_python(os.path.join(data_path,name))
    Mask = m1+m2

    newshape = np.round(np.array(Mask.shape)*spacing/resolution)
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    extendbox = extendbox.astype('int')



    convex_mask = m1
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1+dm2
    Mask = m1+m2
    extramask = dilatedMask - Mask
    bone_thresh = 210
    pad_value = 170
    im[np.isnan(im)]=-2000
    sliceim = lumTrans(im)
    sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = sliceim*extramask>bone_thresh
    sliceim[bones] = pad_value
    sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
    sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                extendbox[1,0]:extendbox[1,1],
                extendbox[2,0]:extendbox[2,1]]
    sliceim = sliceim2[np.newaxis,...]
    np.save(os.path.join(prep_folder,name+'_clean.npy'),sliceim)


    if len(label)==0:
        label2 = np.array([[0,0,0,0]])
    elif len(label[0])==0:
        label2 = np.array([[0,0,0,0]])
    elif label[0][0]==0:
        label2 = np.array([[0,0,0,0]])
    else:
        haslabel = 1
        label2 = np.copy(label).T
        label2[:3] = label2[:3][[0,2,1]]
        label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        label2[3] = label2[3]*spacing[1]/resolution[1]
        label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
        label2 = label2[:4].T
    np.save(os.path.join(prep_folder,name+'_label.npy'),label2)

    print(name)


def full_prep(step1=True,step2 = True):
    warnings.filterwarnings("ignore")

    #preprocess_result_path = './prep_result'
    prep_folder = config['preprocess_result_path']
    data_path = config['stage1_data_path']
    finished_flag = '.flag_prepkaggle'

    if not os.path.exists(finished_flag):
        alllabelfiles = config['stage1_annos_path']
        tmp = []
        for f in alllabelfiles:
            content = np.array(pandas.read_csv(f))
            content = content[content[:,0]!=np.nan]
            tmp.append(content[:,:5])
        alllabel = np.concatenate(tmp,0)
        filelist = os.listdir(config['stage1_data_path'])

        if not os.path.exists(prep_folder):
            os.mkdir(prep_folder)
        #eng.addpath('preprocessing/',nargout=0)

        print('starting preprocessing')
        pool = Pool()
        filelist = [f for f in os.listdir(data_path)]
        partial_savenpy = partial(savenpy,annos= alllabel,filelist=filelist,data_path=data_path,prep_folder=prep_folder )

        N = len(filelist)
            #savenpy(1)
        _=pool.map(partial_savenpy,range(N))
        pool.close()
        pool.join()
        print('end preprocessing')
    f= open(finished_flag,"w+")

def savenpy_luna(id,annos,filelist,luna_segment,luna_data,savepath):#将原始luna数据转换成numpy数组：分割出肺部然后保存为numpy数组，每个肺部扫描对应的结节也保存为numpy数组
    islabel = True
    isClean = True
    resolution = np.array([1,1,1])
#     resolution = np.array([2,2,2])
    name = filelist[id]
    
    Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
    if isflip:
        Mask = Mask[:,::-1,::-1]# 翻转
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')# resize到单个像素的长宽高为1*1*1毫米
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T # 整个肺部的bounding box。

    this_annos = np.copy(annos[annos[:,0]==int(name)]) # 当前CT文件的所有结节的中心点坐标、直径

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data,name+'.mhd'))
        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('flip!')
        sliceim = lumTrans(sliceim) # 像素值映射到0-256
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8') # 对肺外部区域（此时为0）以170填充每个像素点
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value # 对骨头也填充170
        
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1) #对原始肺部图片重采样，使得单个像素的长宽高为1*1*1毫米
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        np.save(os.path.join(savepath,name+'_clean.npy'),sliceim) # 将肺部抠出来，然后保存，肺部以外的部位用170填充。（因为只是将肺部的图像保存起来，所以结节的中心点位置也要改成在肺部内的相对位置）


    if islabel:

        this_annos = np.copy(annos[annos[:,0]==int(name)])
        label = []
        if len(this_annos)>0:
            
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
            
        label = np.array(label) # 保存了当前CT文件的每个结节的三维中心点坐标和直径
        if len(label)==0:
            label2 = np.array([[0,0,0,0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            label2[3] = label2[3]*spacing[1]/resolution[1]
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1) # 结节的中心点位置要改成在肺部内的相对位置。
            label2 = label2[:4].T
        np.save(os.path.join(savepath,name+'_label.npy'),label2) # 保存本文件对应的所有结节中心点坐标和直径
        
    print(name)

def preprocess_luna(): # 函数功能：将luna_data目录里的所有数据抠出肺部区域并保存为'id_clean.npy'，对应的标签为'id_label.npy'
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag):
        filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd') ]
        annos = np.array(pandas.read_csv(luna_label))

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        
        pool = Pool()
        partial_savenpy_luna = partial(savenpy_luna,annos=annos,filelist=filelist,
                                       luna_segment=luna_segment,luna_data=luna_data,savepath=savepath)

        N = len(filelist)
        #savenpy(1)
        _=pool.map(partial_savenpy_luna,range(N))
        pool.close()
        pool.join()
    print('end preprocessing luna')
    f= open(finished_flag,"w+")
    
def prepare_luna():
# 函数功能：对luna_raw文件夹下的所有子文件夹中的luna原始数据重命名为'id加后缀'然后统一转移到luna_data目录下。肺分割文件也重命名，但不移动。
# 这样，所有的Luna数据都存到luna_data目录下了，所有的肺分割文件都存在luna_segment目录下了，且原始数据的文件名和肺分割文件名一一对应。
    print('start changing luna name')
    luna_raw = config['luna_raw']
    luna_abbr = config['luna_abbr']
    luna_data = config['luna_data']
    luna_segment = config['luna_segment']
    finished_flag = '.flag_prepareluna'
    
    if not os.path.exists(finished_flag):

        subsetdirs = [os.path.join(luna_raw,f) for f in os.listdir(luna_raw) if f.startswith('subset') and os.path.isdir(os.path.join(luna_raw,f))]
        if not os.path.exists(luna_data):
            os.mkdir(luna_data)

#         allnames = []
#         for d in subsetdirs:
#             files = os.listdir(d)
#             names = [f[:-4] for f in files if f.endswith('mhd')]
#             allnames = allnames + names
#         allnames = np.array(allnames)
#         allnames = np.sort(allnames)

#         ids = np.arange(len(allnames)).astype('str')
#         ids = np.array(['0'*(3-len(n))+n for n in ids])
#         pds = pandas.DataFrame(np.array([ids,allnames]).T)
#         namelist = list(allnames)
        
        abbrevs = np.array(pandas.read_csv(config['luna_abbr'],header=None))
        namelist = list(abbrevs[:,1])
        ids = abbrevs[:,0]
        
        for d in subsetdirs:
            files = os.listdir(d)
            files.sort()
            for f in files:
                name = f[:-4]
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id) # 文件名为3位整数值id，如果id只有1位或2位，则前面用0补全
                shutil.move(os.path.join(d,f),os.path.join(luna_data,filename+f[-4:])) # 将文件移动到allset中，并重命名为id加原文件后缀（.mhd或.raw）
                print(os.path.join(luna_data,str(id)+f[-4:]))

        files = [f for f in os.listdir(luna_data) if f.endswith('mhd')] # 将重命名后的zraw文件名写入mhd头文件中
        for file in files:
            with open(os.path.join(luna_data,file),'r') as f:
                content = f.readlines()
                id = file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.raw\n'
                print(content[-1])
            with open(os.path.join(luna_data,file),'w') as f:
                f.writelines(content)

                
        seglist = os.listdir(luna_segment)
        print('seglist = ',seglist)
        for f in seglist:
            if f.endswith('.mhd'):

                name = f[:-4]
                lastfix = f[-4:]
            else:
                name = f[:-5]
                lastfix = f[-5:]
            if name in namelist:
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)

                shutil.move(os.path.join(luna_segment,f),os.path.join(luna_segment,filename+lastfix)) # 重命名当前segment file为'id.mhd'和'id.zraw'
                print(os.path.join(luna_segment,filename+lastfix))


        files = [f for f in os.listdir(luna_segment) if f.endswith('mhd')] # 将重命名后的zraw文件名写入mhd头文件中
        for file in files:
            with open(os.path.join(luna_segment,file),'r') as f:
                content = f.readlines()
                id =  file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.zraw\n'
                print(content[-1])
            with open(os.path.join(luna_segment,file),'w') as f:
                f.writelines(content)
    print('end changing luna name')
    f= open(finished_flag,"w+")
    
if __name__=='__main__':
    #full_prep(step1=True,step2=True)
    prepare_luna()
    preprocess_luna()
    
