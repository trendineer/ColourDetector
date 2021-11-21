import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image

colorArray = [
  "LIGHT BLUE",
  "DARK BLUE",
  "LIGHT RED",
  "DARK RED",
  "LIGHT GREEN",
  "DARK GREEN",
  "ORANGE",
  "BROWN",
  "PURPLE",
  "PINK",
  "LIGHT YELLOW",
  "DARK YELLOW",
]

color_to_check = np.ones([200,200,3], dtype='int8') * np.asarray([3,3,3])
print(np.asarray(color_to_check))
# img = cv2.imread(color_to_check)
# im = Image.fromarray(color_to_check)
cv2.imshow('yo',np.asarray(color_to_check))
cv2.waitKey(0)
# print(color_to_check)

# layout = [
#   [[sg.Text("Pick a colour"), sg.Image(background_colour = (2,55,255)) ]]
# ]

# for color in colorArray:
#   layout.append([sg.Button(color)])

# print(layout)

# # layout = [[sg.Text("Hello from PySimpleGUI")], [sg.Button("Light Red")],[sg.Button("Dark Red")],[sg.Button("Light Blue")],[sg.Button("Dark Blue")]]

# window = sg.Window("Yo", layout,  margins=(200, 10))


# def eventColor(event, color):
#   if event == color or event == sg.WIN_CLOSED:
#     return color, True

# while True:
#   event, values = window.read()

#   for color in colorArray:

#     instance = eventColor(event, color)
#     print(instance[0])
#     if (instance[1] == 1):
#       break
  

  
  
#   break
  # if event == "LIGHT RED" or event == sg.WIN_CLOSED:
  #     print('LIGHT RED')
  #     # break
  # if event == "DARK RED" or event == sg.WIN_CLOSED:
  #     print('DARK RED')
  #     # break
  # if event == "LIGHT BLUE" or event == sg.WIN_CLOSED:
  #     print('LIGHT BLUE')
  #     # break
  # if event == "DARK BLUE" or event == sg.WIN_CLOSED:
  #     print('DARK BLUE')
  #     break
  # if event == "Dark Red" or event == sg.WIN_CLOSED:
  #     print('Dark Red')
  #     break
  # if event == "Dark Red" or event == sg.WIN_CLOSED:
  #     print('Dark Red')
  #     break
  # if event == "Dark Red" or event == sg.WIN_CLOSED:
  #     print('Dark Red')
  #     break
  # if event == "Dark Red" or event == sg.WIN_CLOSED:
  #     print('Dark Red')
  #     break
  # if event == "Dark Red" or event == sg.WIN_CLOSED:
  #     print('Dark Red')
      # break
# window.close()





# import csv file

# extract colors from each thing
#   into colorlist array

# for color in colorList:
#   display the colour
  
#   display gui for some number of options
#     light blue 
#     dark blue
#     light red 
#     dark red
#     light green 
#     dark green
#     orange
#     brown 
#     purple 
#     pink 
#     light yellow 
#     dark yellow

#   choose the relevant color and save into color array

#   some way to save the data maybe? so that I can start some time and then continue another time?\


