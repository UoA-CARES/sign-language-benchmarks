import os
import glob
import torch
import cv2
from ViTPose.inference import VitPose

# Set visualise = True to see the pose results
visualise = False


def labelpersons(model,path, classes):   
    imgs = glob.glob(path+ os.sep + "*.jpg")
    imgs.sort()
    boxes = []
    for i in imgs:
        img = cv2.imread(i)
        h,w,c = img.shape

        preds = model(img) 

        highestconf = 0
        highestbbox = None
        for p in preds.xyxy[0].cpu().tolist():

            #only keep highest confidence and class
            if(p[4]>highestconf and p[5] in classes):
                highestconf = p[4]
                highestbbox = p[0:4]
        if(highestbbox!= None):    
            boxes.append ([i.split(os.sep)[-1],highestbbox])
        else:
            boxes.append([i.split(os.sep)[-1],[]])
    
    return boxes
def decimal2pixel(xy, w,h):
    x,y = xy
    return [int(x*w), int(y*h)]
def getBbox(points):
    xmax =0
    xmin =1000
    ymax =0
    ymin = 1000
    for point in points:
        y,x = point[0:2]
        if(y>ymax):
            ymax = y
        if(y<ymin):
            ymin = y
        if(x>xmax):
            xmax = x
        if(x<xmin):
            xmin = x
        topleft = [xmin, ymax]
        botright = [xmax, ymin]
    return topleft, botright

paths = ['val', 'train', 'test']

vitpose = VitPose()
cwd = os.getcwd()
yolov5model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
for path in paths: #loop through parent folders ie [train, val, test]
    folders = next(os.walk(path))[1]
    os.chdir('ViTPose')
    for folder in folders: #loop through videos ie [0,1,2,3]
        os.makedirs(cwd + os.sep + path+ "_viz"+os.sep + folder , exist_ok=True)
        try:
            os.remove(cwd + os.sep + path + os.sep + folder + os.sep + "pose.txt")
        except:
            pass
        file = open( cwd + os.sep + path + os.sep + folder + os.sep + "pose.txt","a")#append mode
        ## body bounding box (yolov5)
        allhumanbboxes = labelpersons(yolov5model ,cwd+ os.sep + path+os.sep + folder,[0])
        #expand to largest box so no bounding box jitter
        xmin =100000000
        ymin =100000000
        xmax =0
        ymax =0

        for humanbbox in allhumanbboxes: #find furtherest extents of bounding boxes
            if(len(humanbbox[1])>0):
                x0,x1 = int(humanbbox[1][0]),int(humanbbox[1][2])
                y0,y1 = int(humanbbox[1][1]),int(humanbbox[1][3])
                if(x0<xmin):
                    xmin =x0
                if(y0<ymin):
                    ymin = y0
                if(x1>xmax):
                    xmax = x1
                if(y1>ymax):
                    ymax = y1

        expand = 1.05 # %
        width = xmax-xmin
        height = ymax - ymin
        maxside = max(width , height) * expand    
        centerx = xmin + width/2
        centery = ymin + height/2         
        x0,x1 = int(centerx - maxside/2), int(centerx + maxside/2)
        y0,y1 = int(centery - maxside/2), int(centery + maxside/2)
        largestbox = [x0,x1,y0,y1]

        for bboxn, humanbbox in enumerate(allhumanbboxes ): #loop through images in video folder [frame0, frame1, frame2]
            if(len(humanbbox[1])>0):
                img = cv2.imread(cwd + os.sep + path + os.sep + folder + os.sep + humanbbox[0])     
                  
                crop = img[largestbox[0]:largestbox[1],largestbox[2]:largestbox[3]]#cv2.rectangle(img, [x0,y0],[x1,y1], (100,40,20))
                points = vitpose.inference(crop)    
  
                h,w,c = img.shape
                ## visualise points ##
                for p in points:
                    y,x = int(p[0] ), int(p[1])
                    crop = cv2.circle(crop, (x,y), radius=1, color=(0, 0, 255), thickness=1)

                
            
                ####head bounding box ###
                headwidth= points[4][1] - points[3][1] 
                headleftx, headlefty = points[3][1],points[3][0]
                headrightx, headrigthy = points[4][1],points[4][0]
                offsety = headwidth/1.5
                offsetx = (((offsety * 2) - headwidth)/2)
                headbotleft = [int(headleftx-offsetx),int( headlefty-offsety)]
                headtopright = [int(headrightx+offsetx), int(headrigthy+offsety)]
                crop= cv2.rectangle(crop, headbotleft, headtopright , (100,40,20))

                
                ### hand bounding boxes ###
                leftwrist = points[9][0:2]
                rightwrist = points[10][0:2]
                offset = headwidth*1.3
                leftwristbboxbotleft = [int(leftwrist[1]- offset), int(leftwrist[0]-offset)]
                leftwristbboxtopright = [int(leftwrist[1]+ offset), int(leftwrist[0]+offset)]
                crop = cv2.rectangle(crop, leftwristbboxbotleft, leftwristbboxtopright, (100,40,20))
                rightwristbboxbotleft = [int(rightwrist[1]- offset), int(rightwrist[0]-offset)]
                rightwristbboxtopright = [int(rightwrist[1]+ offset), int(rightwrist[0]+offset)]
                crop = cv2.rectangle(crop, rightwristbboxbotleft, rightwristbboxtopright , (100,40,20))

                
                # write to text file
                if visualise:
                    cv2.imwrite(cwd + os.sep + path+ "_viz"+os.sep + folder + os.sep + humanbbox[0].replace('.jpg','_.jpg'), crop)

                file.write(humanbbox[0] + ' ') #image file name
                #pose coordinates
                for i in points:
                    file.write(',')
                    for j in i:
                        file.write(str(j)+ ' ')
                #head bounding box
                file.write(", " + str(headtopright[1]) + ' ' + str(headtopright[0]))
                file.write(", " + str(headbotleft[1]) + ' ' + str(headbotleft[0]))
                
                #left and right hand bounding box
               
                file.write(", " + str(leftwristbboxtopright[0]) + ' ' + str(leftwristbboxtopright[1]))
                file.write(", " + str(leftwristbboxbotleft [0]) + ' ' + str(leftwristbboxbotleft [1]))
               
                file.write(", " + str(rightwristbboxtopright[0]) + ' ' + str(rightwristbboxtopright[1]))
                file.write(", " + str(rightwristbboxbotleft[0]) + ' ' + str(rightwristbboxbotleft[1]))
                #body bounding box
                file.write(", "  )
                file.write(str(x0)+ " " + str(y0)+ ", "+ str(x1)+ " "+ str(y1)+ " " )

                file.write('\n')
            else: #no human detected
                file.write(humanbbox[0] ) #image file name
                file.write('\n')
        
        file.close()
    os.chdir(cwd)