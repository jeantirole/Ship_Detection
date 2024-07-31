import argparse
import os
import json
from configs import args
from PIL import Image
import aPack_model
import numpy as np
import pandas as pd
import datetime
import utm
import rasterio

import matplotlib.pyplot as plt
import matplotlib.patches as patches





def main(args):
    
    from datetime import datetime
    input_string =  str(args.year_) + str(args.month_) + str(args.day_) + str(args.hour_) +  str(args.min)
    input_format = "%Y%m%d%H%M"
    datetime_object = datetime.strptime(input_string, input_format)
    print("Satellite image time : ",datetime_object)
    
    
    # Draw image 
    Image.MAX_IMAGE_PIXELS = 933120000
    img_path = args.img_path
    img_ = Image.open(img_path)
    print("Satellite image size : ", img_.size)
    
    # model infer 
    if args.model_test:
        result = aPack_model.infer_model(input_file_path=args.img_path,\
                            thres=args.infer_threshold , gpu_id = args.gpu_id)

    # result save
 
    save_root = "/data/00.Data/Inference_results"
    file_name = args.img_path.split("/")[-1].split(".")[0]+".npy"    

    if args.model_test == False:
        result = np.load(os.path.join(save_root,file_name))
    else:
        print("Result Saved")
        np.save(os.path.join(save_root,file_name), result)
    
    #-----------------------------------------------------------------------
    # AIS Csv
    #
    #-----------------------------------------------------------------------
    df = pd.read_csv(args.data_path)
        
    #---- current time ----#
    year_= args.year_
    month_ = args.month_ 
    day_ = args.day_
    hour_ = args.hour_
    min  = args.min
    sec =  args.sec
    crit_time_str = f"{year_}-{month_}-{day_} {hour_}:{min}:{sec}"
    date = pd.to_datetime(crit_time_str)

    #---- time condition ----#
    time_gap_min = args.time_gap_min
    time_gap_sec = args.time_gap_sec

    earl_time = date - datetime.timedelta(minutes=time_gap_min,seconds=time_gap_sec)
    post_time = date + datetime.timedelta(minutes=time_gap_min,seconds=time_gap_sec)

    print("early : ",earl_time)
    print("now : ",date)
    print("late : ",post_time)
    
    df.columns = ['mmsi_', 'time', 'long', 'lat', 'heading_', 'turn_', 'speed_']
    
    
    
    #-----------------------------------------------------------------------
    # filtering datetime
    #
    #-----------------------------------------------------------------------
    time_ = []
    for i,row in df.iterrows():
        time_.append( pd.to_datetime( row["time"] ) )
    df["datetime"] = time_

    con1 = df["datetime"] < post_time 
    con2 = df["datetime"] > earl_time

    #--- exec ---#
    time_filter_flag = True

    if time_filter_flag:
        t_df = df.loc[con1 & con2] 
        df.loc[con1 & con2]
    else:
        t_df = df
        t_df
    
    
    #-----------------------------------------------------------------------
    # Coordinates filtering function ! 
    #
    #-----------------------------------------------------------------------
    
        
    if args.txt_path != None:
        txt_path = args.txt_path
        f = open(txt_path,'r')
        lines = f.readlines()
        lines

        top_left =     lines[1]
        bottom_right = lines[-1]

        min_long = float( top_left.split(":")[1].split(",")[1][0:10] )
        max_long = float( bottom_right.split(":")[1].split(",")[1][0:10] ) 

        min_lat  = float( bottom_right.split(":")[1].split(",")[0][0:10] ) 
        max_lat  = float( top_left.split(":")[1].split(",")[0][0:10] ) 

    elif args.xml_path != None:
        # XML parsing
        import xml.etree.ElementTree as ET

        xml_path = args.xml_path
        tree = ET.parse(args.xml_path)
        root = tree.getroot() 


        Top_left = []
        Bottom_right = []

        for child in root:
            if child.tag =="Image":
                print(child)
                for grand_child in child:
                    if grand_child.tag =="PAN":
                        for gg_child in grand_child:
                            #print(gg_child.tag)
                            if gg_child.tag == "ImagingCoordinates":
                                for ggg in gg_child:
                                    
                                    if ggg.tag =="ImageGeogTL":
                                        # print(ggg.tag)                         
                                        # print(ggg[0].text)
                                        # print(ggg[1].text)
                                        Top_left.append(float(ggg[0].text))
                                        Top_left.append(float(ggg[1].text))
                                    elif ggg.tag == "ImageGeogBR":
                                        # print(ggg.tag)                         
                                        # print(ggg[0].text)
                                        # print(ggg[1].text)
                                        Bottom_right.append(float(ggg[0].text))
                                        Bottom_right.append(float(ggg[1].text))
        
        min_long = Top_left[1]
        max_long = Bottom_right[1]
        min_lat = Bottom_right[0]
        max_lat = Top_left[0]

    print(min_long,max_long,min_lat,max_lat)
    
    
    con1 = t_df["long"] > min_long
    con2 = t_df["long"] < max_long

    con3 = t_df["lat"] > min_lat
    con4 = t_df["lat"] < max_lat + 0.0225 # debugged by eric 


    #------------------------------------
    location_filter_flag = True

    if location_filter_flag == True:
        #q_df = df.loc[con1 & con2 & con3 & con4]
        q_df = t_df.loc[con1 & con2 & con3 & con4]
        q_df
    else:
        q_df = df
    
    
    
    #-----------------------------------------------------------------------
    # utm => pixels  ! 
    #
    #-----------------------------------------------------------------------
    utm_array =[] # long lat 
    #--- csv 
    # latlong --> utm
    for lat,lon in zip(q_df['lat'].values,q_df['long'].values):
        #print(lat)
        # if lat < 80:
        res = utm.from_latlon(lat,lon,52,"N")
        utm_array.append(res)

    q_df['utm_array']= utm_array


    pixels_=[] 
    tf_path = args.tf_path
    with rasterio.open(tf_path) as map_layer:

        for inp in utm_array:
        
            coords2pixels = map_layer.index(inp[0],inp[1]) #input lon,lat # 좌표계 
            pixels_.append(coords2pixels)

    q_df['pixels'] = pixels_

    #--- init index in q_df
    q_df.drop_duplicates(inplace=True)
    q_df.reset_index(inplace=True,drop=True)
        
    
    #-----------------------------------------------------------------------
    # AIS matching with Bbox
    #
    #-----------------------------------------------------------------------
    
    # Draw image 
    img_path = args.img_path
    img_ = Image.open(img_path)
    plt.figure(figsize=(18,18))
    #plt.imshow(img_)
    ax = plt.gca()
        
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


    # 데이터들은 q_df 써서 활용하는 것을 원칙으로 한다. 
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
        path_obj = patches.Rectangle((x-width*0.5,y-height*0.5), width, height, 
                                    linewidth=0.1, edgecolor="yellow", fill=False,
                                    rotation_point="center",
                                    angle=angle_*180/np.pi,
                                    alpha=0.35)
        
        ax.add_patch(path_obj)


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

    
    #----------------------
        


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
    
    
    
    #-----------------------------------------------------------------------
    # Masking BBox in the land 
    #
    #-----------------------------------------------------------------------
    
    print(args.map_path)
    os.path.exists(args.map_path)
        
    
    
    return 0 