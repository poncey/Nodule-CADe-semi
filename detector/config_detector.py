# coding=utf-8
config = {#'stage1_data_path':'/work/DataBowl3/stage1/stage1/',
          'luna_raw':'/home/user/work/DataBowl3/data/luna/raw/',
          'luna_segment':'/home/user/work/DataBowl3/data/luna/raw/seg-lungs-LUNA16/',
          
          'luna_data':'/home/user/work/DataBowl3/data/luna/allset/',
          'preprocess_result_path':'/home/user/work/DataBowl3/data/luna/preprocessed_luna_data/',       
          
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
