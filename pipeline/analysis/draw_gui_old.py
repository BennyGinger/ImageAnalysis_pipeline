from __future__ import annotations
import cv2
import numpy as np
from platform import system
from pipeline.utilities.data_utility import load_stack, create_save_folder, save_tif
from pipeline.mask_transformation.complete_track import complete_track
from os import PathLike, scandir, sep
from os.path import join
from skimage.color import gray2rgb
from skimage.draw import polygon2mask



def draw_polygons(img, frames):
    """
    Free-hand draw a polygon on loaded image. Take as input list of path files of the images.
    Usage:
        - The image will appear in popup window
        - Press either 'b' or 'f' to move frames back- or forward
        - Draw polygon
        - Press 's' to save polygon. If a polygon is drawn and save on the same frame, it will overwrite the previous one.
        - Press 'q' when ready

    Args:
        img_list ([str]): List of path file of the images.
        
    Returns:
        dict_roi (dict): Dictionary containg the coordinates of the polygons drawn for each selected frames. 
    """

    seqLeng = frames
    # Load images and draw
    f = 0 # Allow to move between the different frames
    dict_roi = {} # Store the polygons
    img2  = img.copy()
    alpha = 1; beta = 0
    togglemask = 0; togglelabel = 0
    conbri = 0
    while f!=-1:
        drawing=False; polygons = []; currentPt = []
        # Mouse callback function
        def freehand_draw(event,x,y,flags,param):
            nonlocal polygons, drawing, im, currentPt
            # Press mouse
            if event==cv2.EVENT_LBUTTONDOWN:
                drawing=True; polygons = []; currentPt = []
                currentPt.append([x,y])
                polygons = np.array([currentPt], np.int32)
                im = im2.copy()
            # Draw when mouse move, if pressed
            elif event==cv2.EVENT_MOUSEMOVE:
                if drawing==True:
                    cv2.polylines(im,[polygons],False,(0,255,255),2)
                    currentPt.append([x,y])
                    polygons = np.array([currentPt], np.int32)
            # Release mouse button
            elif event==cv2.EVENT_LBUTTONUP:
                drawing=False
                cv2.polylines(im,[polygons],True,(0,255,255),2)
                cv2.fillPoly(im,[polygons],(0,255,255))
                currentPt.append([x,y])
                polygons = np.array([currentPt], np.int32)
            return polygons
        
        # Read/Load image
        if seqLeng==1:
            im = img.copy() #cv2.resize(img,(1000,1000),cv2.INTER_NEAREST)
        else:
            im = img[f].copy() #cv2.resize(img[f],(1000,1000),cv2.INTER_NEAREST) #768
        im2 = im.copy()
        if togglemask == 0:
            if f in dict_roi.keys() and not drawing:
                cv2.polylines(im,dict_roi[f], True, (0,255,255),2)
                cv2.fillPoly(im,dict_roi[f],(0,255,255))
                polygons = dict_roi[f]

        cv2.namedWindow("Draw ROI of the Wound", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Draw ROI of the Wound",freehand_draw)
        cv2.resizeWindow("Draw ROI of the Wound", 1024, 1024)
        
        # Setup labels
        labels = {
    "texts": {"text": "Press 's' to save ROI and move forward", "coord": (10,40)},
    "textar": {"text": "Press 'l' to go forward" if system()=='Linux' 
               else "Press 'ARROW RIGHT' to go forward", "coord": (10,60)},
    "textal": {"text": "Press 'j' to go backward" if system()=='Linux' 
               else "Press 'ARROW LEFT' to go backward", "coord": (10,80)},
    "textq": {"text": "Press 'q' for quit", "coord": (10,100)},
    "textc": {"text": "Press 'c' once and 'i'(up) or 'k'(down) to change contrast" if system()=='Linux' 
              else "Press 'c' once and 'ARROW UP/DOWN' to change contrast", "coord": (10,120)},
    "textb": {"text": "Press 'b' once and 'i' or 'k' to change brightness" if system()=='Linux' 
              else "Press 'b' once and 'ARROW UP/DOWN' to change brightness", "coord": (10,140)},
    "textx": {"text": "Press 'x' to toggle mask", "coord": (10,160)},
    "textl": {"text": "Press 'h' to toggle help", "coord": (10,180)}}
        
        font = cv2.FONT_HERSHEY_PLAIN; fontScale = 1.2; color = (0,255,255); thickness = 1
        
        # Apply label on images
        cv2.putText(im, f"Frame {f+1}/{seqLeng}", (320,20), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(im2, f"Frame {f+1}/{seqLeng}", (320,20), font, fontScale, color, thickness, cv2.LINE_AA)
        
        for label in labels.values():
            if togglelabel == 0:
                cv2.putText(im, label["text"], label["coord"], font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(im2, label["text"], label["coord"], font, fontScale, color, thickness, cv2.LINE_AA)
        
        while True:
            cv2.imshow("Draw ROI of the Wound",im)

            # Numbers for arrow keys
            # Windows: left: 2424832, right: 2555904, up: 2490368, down: 2621440
            # Linux: left: 65361, right: 65363, up: 65362, down: 65364
            
            key = cv2.waitKeyEx(1) 
            # if key != -1:
            #     print(f'{key=}')
            
            # press 'q' to exit.
            if key == ord("q"):
                f = -1
                conbri = 0
                break
            # press 'arrow key right' to move forward
            elif key == 2555904 or key == ord("l") or key==65363: #ArrowKey RIGHT for Windows
                conbri = 0
                if f == seqLeng-1:
                    f = seqLeng-1
                    break
                else:
                    f += 1
                    break
            # press 'arrow key left' to move backwards
            elif key == 2424832 or key == ord("j") or key==65361: #ArrowKey LEFT for Windows
                conbri = 0
                if f == 0:
                    f = 0
                    break
                else:
                    f -= 1
                    break
            # press 's' to save roi.
            elif key == ord("s"):
                conbri = 0
                if polygons.size > 0:
                    if f == seqLeng-1:
                        dict_roi[f] = polygons
                        f = seqLeng-1
                        break
                    else:
                        dict_roi[f] = polygons
                        f += 1
                        break   
            # press 'c' to activate contrast change mode
            elif key == ord("c"):
                conbri = key
            # press 'b' to activate brightness change mode
            elif key == ord("b"):
                conbri = key
            # if Arrowkey up or down is pressed, check if contrast or brightness change mode is active
            elif key == 2490368 or key == ord("i") or key==65362: #ArrowKey UP for Windows
                if conbri == ord("c"):
                    alpha += 5
                    img = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)
                    break
                elif conbri == ord("b"):
                    beta += 5
                    img = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)  
                    break 
            elif key ==  2621440 or key == ord("k") or key==65364: #ArrowKey DOWN for Windows
                if conbri == ord("c"):
                    alpha += -5
                    if alpha < 1:
                        alpha = 1
                    img = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)
                    break
                elif conbri == ord("b"):
                    beta += -5
                    img = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)
                    break   
            # toogle masks visibility
            elif key == ord("x"):
                if togglemask == 0:
                    togglemask = 1
                    break
                else:
                    togglemask = 0
                    break
            # toogle label visibility
            elif key == ord("h"):
                if togglelabel == 0:
                    togglelabel = 1
                    break
                else:
                    togglelabel = 0     
                    break
    cv2.destroyAllWindows()
    return dict_roi

def polygon_into_mask(frames: int, poly_dict:dict, img_shape: tuple[int,int])->np.array:
    mask_stack = np.zeros((frames,*img_shape),dtype=('uint8'))
    for frame, polygon in poly_dict.items():
        # Invert the x,y to y,x
        polygon = [[y,x] for x,y in polygon[0]]
        tempmask =  polygon2mask(img_shape,polygon).astype('uint8')
        mask_stack[frame] = tempmask
    return mask_stack

