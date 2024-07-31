import os
import rasterio    
import utm
import geopandas as gpd
from shapely.geometry import Point
import numpy as np 


def masking_bbox(args, M_):
    
    
    '''
    해안 마스킹 파일에 따라서, bbox 좌표를 하나씩 필터링. 
    
    
    '''
    print(args.map_path)
    print(os.path.exists(args.map_path))

    # Visualize Shape File !!
    import geopandas as gpd
    from shapely.geometry import Point
    # Masking을 위한 shape 불러오기 
    gdf=gpd.GeoDataFrame.from_file(args.map_path) #shapefile 불러오기

    if args.mask_on == True:
        gdf['geometry']= gdf.geometry.to_crs('EPSG:4326')

    # from pixel to utm
    tmp_ =[] 
    with rasterio.open(args.tf_path) as map_layer:
        for x,y in zip(M_['bbox x coord'].values, M_['bbox y coord'].values):
            
            x_coord,y_coord = map_layer.xy(y,x) # 여기 순서에 맞춰줘야, long,lat 좌표계가 맞음 
            tmp_.append([x_coord,y_coord])

    tmp_1= []
    ship_idxs = []
    # from utm to latlon 
    for idx,co in enumerate(tmp_):
        x,y = co[0],co[1]
        lat,lon = utm.to_latlon(x,y,52,"N")
        # print(lat,lon) # M_ 에서, 미리 설정되었던 AIS와 함께 좌표값 검증하였음

        print("progress : ", idx, " / ",len(tmp_))
        check=gdf['geometry'].contains(Point(lon,lat)) #특정 좌표 (long,lat)이 shapefile 다각형에 포함되는지 확인하기
        checkt=np.where(check == True)[0]
        
        if len(checkt) == 0:
            tmp_1.append(checkt)
            ship_idxs.append(idx)
    
    # visualize the result 
    M_masked = M_.iloc[ship_idxs]
    del M_masked['box index']
    # sav the csv 
    M_masked['box index'] = [i for i in range(len(M_masked))]
    M_masked = M_masked.reset_index(drop=True)
    #M_masked.to_csv(args.csv_output_masked)
    #display(M_masked)
        
    return M_masked