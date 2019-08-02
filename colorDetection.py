print("loading")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
def isSkin(r,g,b,h,s,v):
    condicion1 = False
    condicion2 = False
    if(r>220 and g>210 and b>170 and abs(r-g)>15 and r>b and g>b):
        condicion1 = True
    if(r>95 and g>40 and b>20 and max(r,g,b)-min(r,g,b)>15 and abs(r-g)>15 and r>g and r>b):
        condicion1 = True
    if(h>=0 and h<=50):
        if(v>0.35 and s>0.2):
            condicion2 = True
    if(h>=340 and h<=360):
        if(v>0.35 and s>0.2):
            condicion2 = True
    if(condicion1 == True and condicion2 == True):
        return True
    else:
        return False

def RGBToHSV(r,g,b):
    Hue = 0
    Saturacion = 0
    Value = 0
    rPrima = float(r/255)
    gPrima = float(g/255)
    bPrima = float(b/255)
    Cmax = max(rPrima,gPrima,bPrima)
    Cmin = min(rPrima,gPrima,bPrima)
    delta = Cmax - Cmin
    if(delta == 0):
        Hue = 0
    elif(Cmax == rPrima):
        Hue = 60*(((gPrima-bPrima)/delta)%6)
    elif(Cmax == gPrima):
        Hue = 60*(((bPrima-rPrima)/delta)+2)
    elif(Cmax == bPrima):
        Hue = 60*(((rPrima-gPrima)/delta)+4)    
    if(Cmax == 0):
        Saturacion = 0
    else:
        Saturacion = 1 - (Cmin/Cmax)
    Value = Cmax
    return Hue, Saturacion, Value

def skinDetection(imagen):
    #leer la imagen
    img = cv2.imread(imagen)
    #escalar la imagen
    img = cv2.resize(img,(640,480),interpolation = cv2.INTER_CUBIC)
    #convertir de BGR a LAB
    imgLab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    #separar la imagen en canales L,A, y B
    l_chanel, a_chanel, b_chanel = cv2.split(imgLab)
    #aplicar ecualizacion de histograma de contraste adaptivo limitado
    contrastEqu = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32,32))
    cl = contrastEqu.apply(l_chanel)
    #unir los canales de LAB 
    chanels_unidos = cv2.merge((cl,a_chanel,b_chanel))
    #regresar la imagen a BGR
    final_img = cv2.cvtColor(chanels_unidos,cv2.COLOR_LAB2BGR)
    res = np.hstack((img,final_img)) #stacking images side-by-side
    cv2.imwrite("Resultado.jpg",res)
    #crear nueva imagen en blanco
    newImg = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    cv2.rectangle(newImg,(0,0),(img.shape[1],img.shape[0]),(255,255,255),-1)
    skinmap = newImg
    for x in range(final_img.shape[1]):
        for y in range(final_img.shape[0]):
            b, g, r = final_img[y,x]
            h, s, v = RGBToHSV(r,g,b)
            if(isSkin(r,g,b,h,s,v) == True):
                skinmap.itemset((y,x,0),190)
                skinmap.itemset((y,x,1),190)
                skinmap.itemset((y,x,2),190)
            else:
                skinmap.itemset((y,x,0),255)
                skinmap.itemset((y,x,1),255)
                skinmap.itemset((y,x,2),255)
    cv2.imwrite("skinmapCASIFINAL.jpg",skinmap)
    kernel = np.ones((7,7),np.uint8)
    skinmapFinal = cv2.morphologyEx(skinmap, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("skinmap.jpg",skinmapFinal)
    return skinmapFinal

#para cargar la imagen
imagen = "imagen2.jpg"
skinmap = skinDetection(imagen)
cv2.imshow("skinmap",skinmap)