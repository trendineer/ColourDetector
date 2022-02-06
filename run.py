import cv2
import os
from matplotlib.pyplot import contour
import numpy as np
import pandas as pd
import time
from skimage.draw import polygon
from shapely import geometry
import json

# class contours:
#   def __init__(self):
#     self.contour_list = contour_list
#     sel

#function to calculate minimum distance from all colors and get the most matching color
def getColorName(R,G,B):
  minimum = 10000
  for i in range(len(csv)):
    d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
    if(d<=minimum):
      minimum = d
      cname = csv.loc[i,"color_name"]
  return cname

#function to get x,y coordinates of mouse double click
def draw_function(event, x,y,flags,param):
  if event == cv2.EVENT_LBUTTONDBLCLK:
    global b,g,r,xpos,ypos, clicked
    clicked = True
    xpos = x
    ypos = y
    b,g,r = img[y,x]
    b = int(b)
    g = int(g)
    r = int(r)
  # if event == cv2.EVENT_RBUTTONDBLCLK:
  #     clicked = True
  #     print('yo')

def average_colour(img):
  print(img)
  B, G, R = np.split(img,3)
    
  avgR = int(np.array(R).mean())
  avgG = int(np.array(G).mean())
  avgB = int(np.array(B).mean())

  return getColorName(avgR, avgG, avgB)

def edge_detector(img):
  # use edge detector to identify the different then use contouriong to get unique blobs
  edges = cv2.Canny(img,100,200)
  return edges

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def getContourArray(image, edge_image):
  thresh = cv2.threshold(edge_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  contour = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = contour[0]
  poly_coords = np.array([[]])

  for i in range(len(cnts)):
    poly_coord = np.array([[]]) # should look like np.array(((x1,y1),(x2,y2),(x3,y3)) 
    cnt = cnts[i]
    # np.append(poly_coord, [[]])

    for current_cnt in cnt:
      cnt_to_add = current_cnt[0]
      if poly_coord.size == 0 :
        poly_coord = np.append(poly_coord,(cnt_to_add))
      else:
        poly_coord = np.vstack((poly_coord,cnt_to_add))
  
    # if poly_coords.size == 0 :
    #     poly_coords = np.append(poly_coords,(poly_coord))
    # else:


    # NEED TO RETURN ARRAY OF CONTOUR POINTS 
    # v below stuff can be in another function that takes all the array data and reformats it

|# would like ot create an array in form below
  # array = '''{contours :
  #             "id": id,
  #             "avergae_colour": colour
  #             "central_coords": [coords_x, coords_y],
  #             "contour_coords": ["apple", "banana", "orange"]
  #           }'''

  # data  = json.loads(array)
  # fruits_list = data['fruits']
  # print(data)

  # return (poly_coords)

  # return 
  # [
  #   central_coords: 
  #     [
  #       x : coord_x,
  #       y : coord_y
  #     ]
  #   contour_list:
  #     [
  #       1: [contour_list_1],
  #       2: [contour_list_2],
  #       3: [contour_list_3],
  #       4: [contour_list_4]... 
  #     ]
  # ]
    

# https://scikit-image.org/docs/dev/auto_examples/edges/plot_shapes.html#sphx-glr-auto-examples-edges-plot-shapes-py
# want to use skimage.draw.polygon(r, c[, shape])
# usage   rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
#         img[rr, cc, 1] = 1


  # for cnt in cnts:
  #   #define size of contour 
  #   # if cv2.countourArea(cnt) >= 100:

  #     #size of a contoured section
  #     pixelArea = len(cnt)
  #     # print(cnt)

  #     #create empty array for colour
  #     colour = np.array([],dtype='int32')
  #     # print(cnt[0])

  #     # //need to some somehow pull the colours from the pixel and then find the overall avaeral colour
  #     # fill the cnt 
  #     for px in cnt:
  #       px = px[0][0]
  #       # print(px)
  #       # px_colour = np.array(image[px[0]][px[1]], dtype='int32')
  #       # colour = np.append(colour, px_colour)
  
  # cv2.fillPoly(image, cnts, [13,200,155])

      
      # https://numpy.org/doc/stable/reference/generated/numpy.union1d.html union
      # 
      

        # print(px_colour)
      # # np.resize(colour,(3,32))
      # colour = colour.reshape(-1,3)
      # avg_colour = np.mean(colour,axis = 0, dtype='int32')

      #   # colour_reshape = colour.ravel()
      # print(avg_colour)

        # print(colour_reshape)
        # print(np.mean(cnt, axis=0))
      # print(average_colour(avg_colour))
        # print(np.mean(cnt, axis=0))

    #finding the avargae colour in the contoured section
    #determine the colour of the section
    # cv2.fillPoly(image, cnts, [13,200,155])

def fill(image):
  x,y,w,h = cv2.boundingRect(image)
  cv2.floodFill(image,None,(int(x+w/2),int(y+h/2)),255)
  return 255-image


if __name__ == "__main__":

  # load in image data
  img_path = os.path.abspath(os.getcwd()) + "\\polka-dot.jpg"
  img_path = "polka-dot.jpg"
  img = cv2.imread(img_path)
  img2 = cv2.imread(img_path)
  
  #Reading csv file giving names to each column
  index=["color","color_name","R","G","B"]
  csv = pd.read_csv('colors2.csv', names=index, header=None)

  # Image processing 
  # applying an edge detector
  edge_img = auto_canny(img)  
  curr_img = getContourArray(img, edge_img)
  print(curr_img)


