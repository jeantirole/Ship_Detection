from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 



def vis(args,M_masked):
    
    '''
    1. 마스킹된 bbox를 image 상에 표시하고 저장. 저장되는 위치는 args.img_output_masked
    
    
    '''

    # visualize the result with masking 
    # original image 
    img_path = args.img_path
    img_ = Image.open(img_path)

    plt.figure(figsize=(18,18))
    plt.imshow(img_)
    ax = plt.gca()

    #------------------------------------
    # Bbox 
    bboxes_ = [] 
    cnt = 0
    box_cnt = 0

    # Label
    plt_txts = []

    # M_masked
    for i,row in M_masked.iterrows():
        box_idx = row['box index']
        x = row['bbox x coord']
        y = row['bbox y coord']
        width = row['bbox width']
        height = row['bbox height']
        angle_ = row['bbox angle']
        prob_ = row['bbox prob']

        mmsi_ = row['mmsi']
        heading_ = row['heading']
        turn_ = row['turn']
        speed_ = row['speed']
        lat_ = row['ais lat']
        lon_ = row['ais long']

        
        
        box_cnt+=1
        print(f"#-------------- {box_cnt / len(M_masked):.4f}")
        path_obj = patches.Rectangle((x-width*0.5,y-height*0.5), width, height, linewidth=0.1, edgecolor="red", fill=False,
                                    rotation_point="center",
                                    angle=angle_*180/np.pi)
        
        ax.add_patch(path_obj)

        # if (lat_ ==0) & (lon_ ==0):
        #     pass
        # else:
        name = f'bbox idx : {box_idx} mmsi : {mmsi_} heading : {heading_} turn : {turn_} speed : {speed_} ais coord: {lon_,lat_} bbox prob : {prob_}'
        #plt_txts.append( plt.text(x, y, name, fontsize =2,color='red') ) 


    # adjust_text(plt_txts, 
    #             force_points=3,
    #             arrowprops={'arrowstyle' : '->', 
    #                         'color' : 'crimson',  
    #                         'alpha' : 0.1}
    #             )
    #------------------------------------

    #plt.imshow(img_)
    plt.savefig(args.img_output_masked,dpi =800 )
    #plt.show()
    #------------------------------------