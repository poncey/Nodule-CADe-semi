#coding=utf-8

import numpy as np
import pandas
import SimpleITK as sitk
from glob import glob
import os
from pandas import DataFrame
import numpy as np
import pandas
import SimpleITK as sitk
from glob import glob
import os
from pandas import DataFrame
import csv

def true_or_false_positive(lbb,pbb):#判断通过标签还有预测中心距离是否大于肺结节半径判断该结节的真假阳性
    if lbb.shape[0]==0:
        flag_lbb = None
    else:
        flag_lbb = np.ones(len(lbb),dtype=np.bool)
    flag_pbb = np.zeros(len(pbb),dtype=np.bool)
    for i in range(len(lbb)):
        lbb_x,lbb_y,lbb_z,lbb_d = lbb[i]

        for j in range(len(flag_pbb)):
            pbb_x,pbb_y,pbb_z,pbb_d = pbb[j,1:]

            dx = lbb_x - pbb_x
            dy = lbb_y - pbb_y
            dz = lbb_z - pbb_z

            if dx**2+dy**2+dz**2<=lbb_d:
                flag_lbb[i]=True
                flag_pbb[j]=True
    return flag_lbb,flag_pbb #输出
        

def pbb_th(pbb,th):
    pbb = pbb[pbb[:,0]>th]
    return pbb

def sigmoid(x):#通过sigmoid输出0到1的概率值
    x=1/(1+np.exp(-x))
    return x

def top_n_pbb(pbb,n):
    sorted_ind = np.argsort(pbb[:,0])
    sorted_ind = sorted_ind[::-1]
    sorted_ind = sorted_ind[:n]
    pbb=pbb[sorted_ind]

    return pbb

def iou(box0, box1):

    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection

    return intersection / union

def nms(output, nms_th):
    #print '调用了NMS'
    if len(output) == 0:
        return output

def sigmoid(x):
    x=1/(1+np.exp(-x))
    return x

def top_n_pbb(pbb,n):
    sorted_ind = np.argsort(pbb[:,0])
    sorted_ind = sorted_ind[::-1]
    sorted_ind = sorted_ind[:n]
    pbb=pbb[sorted_ind]
    
    return pbb

def iou(box0, box1):
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    
    return intersection / union
    
def nms(output, nms_th):
    #print '调用了NMS'
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])] # 可信度从大到小排列
    bboxes = [output[0]]
    
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th: # 与置信度更高的box重叠，所以舍弃
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes
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

resolution = np.array([1,1,1])

#annos = np.array(pandas.read_csv('lunaqualified.csv'))

def get_lung_box(file_id):
    #函数功能：返回肺部的bbox的体素坐标值的numpy数组，具体格式为[[min(x),max(x)],[min(y),max(y)],[min(z),max(z)]]
    #通过lung-segment获取lung_box
    Mask,_,spacing,isflip = load_itk_image('/home/user/work/DataBowl3/data/luna/raw/seg-lungs-LUNA16/'+file_id+'.mhd')

    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')# resize到单个像素的长宽高为1*1*1毫米
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2

    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]]) #肺部的bounding box
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1) #肺部的bounding box缩放到单个像素长宽高为1*1*1毫米
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T # 整个肺部加上margin后的bounding box。
    
    return extendbox

def result_to_world_coord(file_id,bbox_voxel_coord): #经过验证，本函数可以将label还原成世界坐标。因此，也可以将pbb还原成世界坐标。
    #file_id是字符串
    #读取需要转换的数据bbox_voxel_coord，数据类型是np.array，形状是(n,4)
    #bbox_voxel_coord = np.load('/home/jx/work/DataBowl3/data/luna/preprocessed_luna_data/'+file_id+'_label.npy')
    if np.sum(bbox_voxel_coord == np.array([[0,0,0,0]])) == 4: #如果标签是空，则跳过
        return
    #获取需要转换的bbox对应的肺部bounding box，以计算世界坐标
    lung_box = get_lung_box(file_id)

    #获取肺部图像的origin
    _,origin,spacing2,_ = load_itk_image('/home/user/work/DataBowl3/data/luna/allset/'+file_id+'.mhd')

    #print 'bbox_voxel_coord = ',bbox_voxel_coord

    #print lung_box[:,0]

    #print origin
    world_coord = np.zeros((len(bbox_voxel_coord),4))

    world_coord[:,:3]=(bbox_voxel_coord[:,:3] + lung_box[:,0]) * resolution + origin
    world_coord[:,3]=bbox_voxel_coord[:,3]

    #world_coord[:,:3]=world_coord[:,:3]+lung_box[:,0]*resolution

    #world_coord[:,:3]=world_coord[:,:3]+origin

    #print 'spacing2 =',spacing2
    #经过验证，检测结果中的lbb、pbb和preprocessed_luna_data中label的顺序都是z,y,x,d，单位都是像素。因此这里需要将每个world_coord都转换回x,y,z,d的顺序
    for i,row in enumerate(world_coord):
        world_coord[i,:3]=row[:3][::-1]
    #print file_id,world_coord
    
    return world_coord

def file_id_to_seriesuid(file_id):
    return
    
def save_result_as_cvs(file_id,seriesuid,world_coord):
    columns_name = ['coordX','coordY','coordZ','probability']
    df = DataFrame(world_coord,columns=columns_name)
    #将file_id转换成seriesuid
    
    df.insert(0,'seriesuid',seriesuid)
    
    return df

#file_list=['078','097','101']

#file_list=['026', '066', '276', '325', '559', '608', '756', '830'] # 这是测试结果的文件id和本地电脑的mhd id的交集（因为我要读取mhd中的origin和肺部分割文件的lung_box[:0]）

file_list = glob('/home/user/wuyunheng/DSB2017-master/training/results/res18/voifold2/test/bbox/*pbb.npy')
id_list=[]
#print len(file_list)
for f in file_list:
    id_list.append(os.path.splitext(f)[0].split('_')[0][-3:])

shorter = np.array(pandas.read_csv('/home/user/work/DataBowl3/DSB2017-master/training/detector/labels/shorter.csv',header=None))


#all_df = pandas.read_csv('my_submission.csv')

#id_list = ['026','066','276','325','559','608','756','830']
#np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

total_nodule = 0
missed_nodule = 0

total_pbb = 0
true_positive = 0
false_positive = 0
csvFile=open("detect_post/fig.csv", "wb") 
writer =csv.writer(csvFile)
writer.writerow(["index","x","y","z","d","p"])
for file_id in id_list:
    print 'processing ',file_id
    #读取bbox
    #bbox_voxel_coord = np.load('/home/jx/work/DataBowl3/data/luna/preprocessed_luna_data/'+file_id+'_label.npy')
    lbb_name='/home/user/wuyunheng/DSB2017-master/training/results/res18/voifold2/test/bbox/'+file_id+'_lbb.npy'
    pbb_name='/home/user/wuyunheng/DSB2017-master/training/results/res18/voifold2/test/bbox/'+file_id+'_pbb.npy'
    
    lbb=np.load(lbb_name)
    total_nodule += lbb.shape[0]

    pbb=np.load(pbb_name)
    
    pbb[:,0]=sigmoid(pbb[:,0]) # 因为网络输出的时候没有计算sigmoid，因此在这里要对pbb的置信分数做sigmoid计算
   
    pbb = pbb_th(pbb,0.8)
    print lbb.shape[0],'lbbs, ',pbb.shape[0],' pbbs'

    pbb=nms(pbb,0.125)
    total_pbb += pbb.shape[0]
    print 'after NMS,',pbb.shape[0],' pbbs left.'

    #if lbb.shape[0]==0:
    #    for iii in range(pbb.shape[0]):
    #        print '        ',pbb[iii]

    #before_nms=300
    #pbb=top_n_pbb(pbb,before_nms) # 对 top N 个pbb做 NMS
    
    
    
    #after_nms=30
    #N = after_nms
    #pbb=top_n_pbb(pbb,after_nms) # 对 top N 个pbb做分析

    flag_lbb,flag_pbb = true_or_false_positive(lbb,pbb)
    if flag_lbb is not None:
        missed_nodule += np.sum(1-flag_lbb)
    true_positive += np.sum(flag_pbb)
    false_positive +=np.sum(1-flag_pbb)

    #world_coord=result_to_world_coord(file_id,pbb[:,1:])
    #if lbb.shape[0]!=0:
        #lbb_world_coord = result_to_world_coord(file_id,lbb[:,:]) # lbb的格式为[x,y,z,d]
    #world_coord_and_diam = np.zeros((len(pbb),5))

    #world_coord[:,-1]=pbb[:,0]

    #world_coord_and_diam[:,:3] = world_coord[:,:3]
    #world_coord_and_diam[:,3] = pbb[:,4]
    #world_coord_and_diam[:,-1] = world_coord[:,-1]
    
    #world_coord = world_coord_and_diam #将world_coord的格式从[x,y,z,可能性]改为[x,y,z,d,可能性]
    #world_coord[:,:4]=np.around(world_coord[:,:4],decimals=1)
    #world_coord[:,4] = np.around(world_coord[:,4],decimals=4)
    #xx,yy=np.where(shorter==int(file_id))
    #seriesuid = shorter[xx[0]][-1]

    #assert len(flag_pbb)==len(world_coord)
 

    #print pbb
    pbb_diam = np.zeros((len(pbb),5))
    pbb_diam[:,0]=pbb[:,1]
    pbb_diam[:,1]=pbb[:,2]
    pbb_diam[:,2]=pbb[:,3]
    pbb_diam[:,3]=pbb[:,4]
    pbb_diam[:,-1]=pbb[:,0]
    pbb=pbb_diam
    pbb[:,:4]=np.around(pbb[:,:4],decimals=2)
    pbb[:,4]=np.around(pbb[:,4],decimals=4)
    #xx,yy=np.where(shorter==int(file_id))
    #seriesuid=shorter[xx[0]][-1]
    
    assert len(flag_pbb)==len(pbb)
    

    
    i=[]
    if pbb is not None:
        print file_id       #,seriesuid
        if lbb.shape[0]!=0:
            print 'lbb:'
            for jjj in range(len(lbb)):
                print lbb[jjj],flag_lbb[jjj]
        print 'pbb:'
        for iii in range(len(pbb)):
            print pbb[iii],flag_pbb[iii]
            i=[file_id,pbb[iii][0],pbb[iii][1],pbb[iii][2],pbb[iii][3],pbb[iii][4]]
            writer.writerow(i)
            
        #print world_coord
        #df=save_result_as_cvs(file_id,seriesuid,world_coord)
        #all_df=all_df.append(df,ignore_index=True)
        
        #print all_df
    print '-'*100
csvFile.close()
print 'nodule: total ',total_nodule,' miss',missed_nodule
print 'pbb: total',total_pbb,'true_positive = ',true_positive,'false_positive = ',false_positive
#all_df.to_csv('my_submission.csv',index=False)
        
    


#file_list = glob('res18/bbox/*pbb.npy')
#print len(file_list)
#for f in file_list:
#    file_id = os.path.splitext(f)[0].split('_')[0][-3:]
    
    
    #print f
    #print file_id
    
    #result_to_world_coord(file_id)
    


