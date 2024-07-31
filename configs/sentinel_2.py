#---------------------- Header 
from easydict import EasyDict
import os
import json
args = EasyDict()

#------ image list 
i_1 = "/data/00.Data/AO/0-Pusan-2023-AO/C1_20230116015105_10110_00006119_L1G_PS/0-Pusan-2023-AO_3_C1_20230116015105_10110_00006119_L1G.png"
#------ tif list 
t_1 = "/data/00.Data/AO/0-Pusan-2023-AO/C1_20230116015105_10110_00006119_L1G_PS/C1_20230116015105_10110_00006119_L1G_PRGB_georeferencing_32652.tif"
#------ Coordinates filtering function
txt_path = "/data/00.Data/AO/0-Pusan-2023-AO/C1_20230116015105_10110_00006119_L1G_PS/1RCoordinate.txt"
xml_path = None

#------ gpu
gpu_id = 0
#------ model 
img_path = i_1

print( '/'.join( img_path.split("/")[0:-1]) )
infer_threshold = 0.2
infer_threshold_showing = 0.2
source_root ='/'.join( img_path.split("/")[0:-1])

img_output = img_path.replace(".png" ,"_output_v1_0730.png")
img_output_masked = img_path.replace(".png" ,"_output_v1_0730_Masked.png")
img_output_masked = os.path.join( "/root/Ship_Detection_EO/results", img_path.split("/")[-1])

img_output_scatter = img_path.replace(".png" ,"_output_v1_0730_Scatter.png")

csv_output = img_path.replace(".png","_output_v1_0730.csv")
csv_output_masked =img_path.replace(".png","_masked_v1_0730.csv")
csv_output_scatter = img_path.replace(".png","_scatter_v1_0730.csv")


#------ mask 
mask_on = False
map_path = "/data/00.Data/Shape_Korea_Clipped_240730_new"
#------ TIF 
tf_path = t_1

# time 2023 05 23 04 50 50
year_= int( i_1 .split("/")[-2].split("_")[1][0:4])
month_ = int(i_1 .split("/")[-2].split("_")[1][4:6])
day_ = int(i_1 .split("/")[-2].split("_")[1][6:8])
hour_ = int(i_1 .split("/")[-2].split("_")[1][8:10])
hour_ = hour_ + 9
min  = int(i_1 .split("/")[-2].split("_")[1][10:12]) 
sec =  0
print("time : ", year_, month_, day_, hour_, min)

#------ AIS csv ---------------------------------------------------------------------------
ais_root_1 = "/data/00.Data/AIS-CSV-NTO-AO/1차-3개"
ais_root_2 = "/data/00.Data/AIS-CSV-NTO-AO/2차-11개"

if str(year_) == "2022":
    ais_root = ais_root_1
elif str(year_) == "2023":
    ais_root = ais_root_2

#-------
if len(str(month_)) ==1:
    month_ = str("0") + str(month_)

if len(str(day_)) ==1:
    day_ = str("0") + str(day_)

print(str(year_)+str(month_)+str(day_)+str(hour_)+str("00_dynamic.csv"))
csv_name = str(year_)+str(month_)+str(day_)+str(hour_)+str("00_dynamic.csv")
data_path = os.path.join( ais_root , csv_name)
print(data_path)
if os.path.exists(data_path):
    print("#------------------ : csv exists")
else:
    print("#------------------ : no csv found")

static_data_path = data_path.replace("_dynamic.csv","_static.csv")
final_merge = img_path.replace(".png",".csv")

# time gap
# 시각화 결과물에서 ais 좌표가 해상도가 더 높아지게 된다. 
time_gap_min = 5
time_gap_sec = False

#--------------------------------------------------------------------------------------------

# pixel gap 
# this is for visualization
pixel_gap = 200

# # latlon gap 
# # 매칭되는 박스와 ais 간의 기준 => 늘리면 R_ 데이터프레임에서 매칭 케이스가 늘어나게 된다. 
# latlon_gap = 0.000001

#------ Model Inference 
model_test = True
Box_Matching = True
png_draw_save = False


# Save EasyDict to a file
file_path = os.path.join("/root/Ship_Detection_EO/configs/latest_config.json")
with open(file_path, 'w') as json_file:
    json.dump(args, json_file, indent=4)