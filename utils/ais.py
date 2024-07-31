import pandas as pd
import datetime
import utm
import rasterio

def ais_time_filtering(args):
    
    
    '''
    time filtering ais 
    args.time_gap_min 
    args.time_gap_sec 
    args 의 조건에 따라서 분/초 단위로 ais 데이터를 필터링한다. 
    
    1. 영상이 찍힌시간을 파싱
    2. 찍힌 영상의 시간에 +- time_gap 을 설정하여 ais 데이터를 필터링  
    
    
    '''
    df = pd.read_csv(args.data_path)


    #---- current time ----#
    import datetime
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
    #------------------------------
    earl_time = date - datetime.timedelta(minutes=time_gap_min,seconds=time_gap_sec)
    post_time = date + datetime.timedelta(minutes=time_gap_min,seconds=time_gap_sec)

    print("early : ",earl_time)
    print("now : ",date)
    print("late : ",post_time)


    df.columns = ['mmsi_', 'time', 'long', 'lat', 'heading_', 'turn_', 'speed_']

    #---- filtering datetime ---# 
    # 
    time_ = []
    for i,row in df.iterrows():
        #print(row["time"])
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
        
        
    return t_df


#------------------------

def ais_coordinates_filtering(args,t_df):
    '''
    1. 위의 시간과 동일. 주어진 영상 좌표에 맞춰서 ais 데이터 필터링 기능 
    2. utm => pixels 로 변환하는 기능 
    
    
    '''

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

    # 35.032614576
    # 128.632902263
    # ImageGeogBR
    # 34.852878881
    # 128.904069499

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
    
    
    
    
    #-------------------
    # utm => pixels 
    #------------------
    utm_array =[] # long lat 
    #--- csv 
    # latlong --> utm
    for lat,lon in zip(q_df['lat'].values,q_df['long'].values):
        #print(lat)
        # if lat < 80:
        res = utm.from_latlon(lat,lon,52,"N")
        utm_array.append(res)

    # 전환한 utm_array 를 데이터프레임에 병합 
    q_df['utm_array']= utm_array


    pixels_=[] # utm_array 를 pixels_ 로 변경 
    tf_path = args.tf_path
    with rasterio.open(tf_path) as map_layer:

        for inp in utm_array:
        
            coords2pixels = map_layer.index(inp[0],inp[1]) #input lon,lat # 좌표계 
            pixels_.append(coords2pixels)

    q_df['pixels'] = pixels_

    #--- init index in q_df
    q_df.drop_duplicates(inplace=True)
    q_df.reset_index(inplace=True,drop=True)
    
    return q_df


