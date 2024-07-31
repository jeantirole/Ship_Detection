import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
from adjustText import adjust_text


def matching_box_with_ais(args,result,q_df):
    
    
    '''
    1. BBox matching with AIS 
    2. Visualization => patches 에 대한 주석을 해제해야 함 
    3. matching 된 bbox 와 ais 정보를 담고있는 최종 결과물 M_ 
    
    
    '''
    
    

    # Draw image 
    #img_path = args.img_path
    #img_ = Image.open(img_path)
    #plt.figure(figsize=(18,18))
    #plt.imshow(img_)
    #ax = plt.gca()

    # Bounding Box 
    # Thresholding filtering 
    bboxes_ = [] 
    cnt = 0
    for r in result[0]:
        if r[-1] > args.infer_threshold_showing:
            cnt+=1
            bboxes_.append(r)

    #----------------------------------------------------- 
    # BBox matching with AIS 

    # Matched Bbox
    idx_bboxes = []
    x_bboxes = [] 
    y_bboxes = []
    width_bboxes = []
    height_bboxes = []
    angle_bboxes = []
    prob_bboxes= []

    # Matched AIS 
    x_matched_ais = []
    y_matched_ais = []
    time_matched_ais = []
    idxss_matched_ais = []
    lat_matched_ais = []
    long_matched_ais = []

    mmsi_matched_ais = []
    heading_matched_ais = []
    turn_matched_ais = []
    speed_matched_ais = []


    x_ais = [i[1] for i in q_df["pixels"].values]
    y_ais = [i[0] for i in q_df["pixels"].values]
    lat_ais  = [i for i in q_df['lat'].values]
    long_ais = [i for i in q_df['long'].values]
    mmsi_ = [i for i in q_df['mmsi_'].values]
    heading_ = [i for i in q_df['heading_'].values]
    turn_ = [i for i in q_df['turn_'].values]
    speed_ = [i for i in q_df['speed_'].values]

    idx_ais = [i for i in q_df.index.values]
    # debugged by Eric 
    time_ais = [i for i in q_df['time']]


    for bidx, box_ in enumerate( bboxes_ ):
        x = box_[0]
        y = box_[1]
        width =  box_[2]
        height = box_[3]
        angle_ = box_[4]
        prob_  = box_[5]
        

        # model 의 객체좌표가 center 중심인데, patches.Rectangle 은 좌하단을 기준점으로 보기 때문에. x,y 를 아래와 같이 shift 
        # roatate 할 때, 중심좌표 center 로 잡고, angle radian 변환해서 넣어주기 
        # path_obj = patches.Rectangle((x-width*0.5,y-height*0.5), width, height, 
        #                             linewidth=0.1, edgecolor="yellow", fill=False,
        #                             rotation_point="center",
        #                             angle=angle_*180/np.pi,
        #                             alpha=0.35)
        
        # ax.add_patch(path_obj)


        #--- find matched AIS with each box 
        cnt =0 
        for a,b,ix,t,lat_,long_,mmsi,heading,turn,speed in zip(x_ais,y_ais,idx_ais,time_ais, lat_ais,long_ais, mmsi_,heading_,turn_,speed_):
            
            # 각 bbox 범위안에 들어오는 ais 중에서 첫번쨰 ais 만 리스트에 저장 
            if cnt ==1:
                break

            # if bbox matched with ais 
            if (x - args.pixel_gap < a < x + args.pixel_gap) & (y - args.pixel_gap < b < y + args.pixel_gap) :
                
                idx_bboxes.append(bidx)
                x_bboxes.append(x)
                y_bboxes.append(y)
                width_bboxes.append(width)
                height_bboxes.append(height)
                angle_bboxes.append(angle_)
                prob_bboxes.append(prob_)

                #-- 
                x_matched_ais.append(a)
                y_matched_ais.append(b)
                time_matched_ais.append(t)
                idxss_matched_ais.append(ix)
                lat_matched_ais.append(lat_)
                long_matched_ais.append(long_)

                #-- 
                mmsi_matched_ais.append(mmsi) 
                heading_matched_ais.append(heading) 
                turn_matched_ais.append(turn) 
                speed_matched_ais.append(speed) 

                cnt+=1


    #----------------------- indent over ! 

    if args.Box_Matching  == True:
        T_ = pd.DataFrame({
            "box index" : idx_bboxes,
            "bbox x coord" : x_bboxes,
            "bbox y coord" : y_bboxes,
            "bbox width" : width_bboxes,
            "bbox height" : height_bboxes,
            "bbox angle" : angle_bboxes,
            "bbox prob" : prob_bboxes,
            
            #-- ais
            "mmsi" : mmsi_matched_ais,
            "heading" : heading_matched_ais,
            "turn" : turn_matched_ais,
            "speed" : speed_matched_ais,

            "ais x coord(pixel)" : x_matched_ais,
            "ais y coord(pixel)" : y_matched_ais, 
            "ais lat"   : lat_matched_ais,
            "ais long"  : long_matched_ais,
            "ais time" : time_matched_ais
        })

        T_.drop_duplicates(inplace=True)

        S_ = {
            'box index':[i for i in range(len(bboxes_)) if i not in idx_bboxes],
            'bbox x coord' : [],
            'bbox y coord' : [],
            'bbox width' : [],
            'bbox height' : [],
            'bbox angle' :[],
            'bbox prob' : [],
            "ais x coord(pixel)" : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            "ais y coord(pixel)" : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            "ais lat"   : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            "ais long"  : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            "ais time" : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            }

        not_include_boxes = [i for i in range(len(bboxes_)) if i not in idx_bboxes]

        for a in not_include_boxes:
            box_ = bboxes_[a]
            
            S_['bbox x coord'].append(box_[0])
            S_['bbox y coord'].append(box_[1])
            S_['bbox width'].append(box_[2])
            S_['bbox height'].append(box_[3])
            S_['bbox angle'].append(box_[4])
            S_['bbox prob'].append(box_[5])

        S_ = pd.DataFrame(S_)

        M_ = pd.concat([T_,S_])
        #del M_["box index"]
        M_.reset_index(inplace=True,drop=True)


    return M_