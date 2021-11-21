#programming_fever
import cv2
import numpy as np
import pandas as pd


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


if __name__ == "__main__":

  img_path = "C://Users//omare//Google Drive//Dev//Trendineer//Colour Detection//OpenCV-Projects//color detection//palette.jpg"
  img = cv2.imread(img_path)
  img = cv2.resize(img,(700,500))

  clicked = False
  r = g = b = xpos = ypos = 0

  #Reading csv file with pandas and giving names to each column
  index=["color","color_name","hex","R","G","B"]
  csv = pd.read_csv('colors2.csv', names=index, header=None)

  window_name = 'window'
  cv2.namedWindow(window_name)
  cv2.setMouseCallback(window_name,draw_function)

  cv2.setMouseCallback(window_name,draw_function)
  # cv2.setMouseCallback(window_name,rename_function)

  while(1):
    cv2.imshow(window_name,img)
    if (clicked):
      #cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle 
      cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

      #Creating text string to display( Color name and RGB values )
      text = getColorName(r,g,b) + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)

      #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
      cv2.putText(img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)

      #For very light colours we will display text in black colour
      if(r+g+b>=600):
        cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)

      clicked=False

    if cv2.waitKey(20) & 0xFF ==27:
      break

    cv2.destroyAllWindows()