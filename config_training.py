# coding=utf-8
config = {#'stage1_data_path':'/work/DataBowl3/stage1/stage1/',
          'luna_raw':'/home/user/work/DataBowl3/data/luna/raw/', # 所有luna数据存放的原始位置（包含subset0-9）
          'luna_segment':'/home/user/work/DataBowl3/data/luna/raw/seg-lungs-LUNA16/', # luna_segment存放的原始位置
          
          'luna_data':'/home/user/work/DataBowl3/data/luna/allset/', # 所有luna原始数据prepare操作后集中存放的位置
          'preprocess_result_path':'/home/user/work/DataBowl3/data/luna/preprocessed_luna_data/', # 处理好的只包含肺部的numpy数据和numpy标签
          
          'luna_abbr':'./detector/labels/shorter.csv',# shorter.csv这个文件只包含0-887的编号和seriesuid
          'luna_label':'./detector/labels/lunaqualified.csv',# 0-887的编号和对应的CT文件的肺结节坐标和直径
          #'stage1_annos_path':['./detector/labels/label_job5.csv',
          #      './detector/labels/label_job4_2.csv',
          #      './detector/labels/label_job4_1.csv',
          #      './detector/labels/label_job0.csv',
          #      './detector/labels/label_qualified.csv'],
          'bbox_path':'../detector/results/res18/bbox/',
          'preprocessing_backend':'python'
         }
