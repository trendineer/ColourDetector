import cv2
import os
import numpy as np
import pandas as pd
import time

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

def findfill(image, edge_image):
  thresh = cv2.threshold(edge_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  # print(cnts)
  
  for cnt in cnts:

    #define size of contour 
    # if cv2.countourArea(cnt) >= 100:


      #size of a contoured section
      pixelArea = len(cnt)
      # print(cnt)

      #create empty array for colour
      colour = np.array([],dtype='int32')
      # print(cnt[0])

      # //need to some somehow pull the colours from the pixel and then find the overall avaeral colour
      # for px in cnt:

      # fill the cnt 
      for px in cnt:
        px = px[0][0]
        print(px)
        # px_colour = np.array(image[px[0]][px[1]], dtype='int32')
        # colour = np.append(colour, px_colour)

      
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

  img_path = os.path.abspath(os.getcwd()) + "\\polka-dot.jpg"
  img_path = "polka-dot.jpg"
  img = cv2.imread(img_path)
  img2 = cv2.imread(img_path)

  # img = cv2.resize(img,(700,500))

  # clicked = False
  # r = g = b = xpos = ypos = 0
  
  #Reading csv file with pandas and giving names to each column
  index=["color","color_name","R","G","B"]
  csv = pd.read_csv('colors2.csv', names=index, header=None)


  # window_name = 'window'
  # cv2.namedWindow(window_name)
  # cv2.setMouseCallback(window_name,draw_function)

  # cv2.setMouseCallback(window_name,draw_function)
  # # cv2.setMouseCallback(window_name,rename_function)

  # while(1):
  #   cv2.imshow(window_name,img)
  #   if (clicked):
  #     #cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle 
  #     cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

  #     #Creating text string to display( Color name and RGB values ) 
  #     text = getColorName(r,g,b) + ' R=' + str(r) + ' G='+ str(g) + ' B='+ str(b)

  #     #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
  #     cv2.putText(img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)

  #     # #For very light colours we will display text in black colour
  #     if(r+g+b>=600):
  #       cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)

  #     clicked=False

  #   if cv2.waitKey(20) & 0xFF ==27:
  #     break
  edge_img = auto_canny(img)
  # cnts = cv2.findContours(curr_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # cnts = cnts[0] if len(cnts) == 2 else cnts[1] 
  # print(cnts)
  # cv2.fillPoly(curr_img, cnts, [255,255,255])
  curr_img = findfill(img, edge_img)
  # cv2.imshow('img',img)
  # cv2.imshow('img2',img2)
  # cv2.imshow('edge',curr_img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  # print(average_colour(img))
  cv2.imshow('image',img)
  cv2.waitKey(0)

