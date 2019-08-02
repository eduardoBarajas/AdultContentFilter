print("Cargando")
import cv2
import numpy as np
import math
import csv
import time
import os.path
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import warnings

ORIGINAL_IMG = None
TOTAL_SKIN = 0
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
    print("Deteccion De Piel En Proceso...")
    #leer la imagen
    img = cv2.imread(imagen)
    #escalar la imagen
    img = cv2.resize(img,(640,480),interpolation = cv2.INTER_CUBIC)
    #convertir de BGR a LAB
    imgLab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    #separar la imagen en canales L,A, y B
    l_chanel, a_chanel, b_chanel = cv2.split(imgLab)
    #aplicar ecualizacion de histograma de contraste adaptivo limitado
    contrastEqu = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,4))
    #mejor hasta ahora
    #contrastEqu = cv2.createCLAHE(clipLimit=5, tileGridSize=(1,4))
    cl = contrastEqu.apply(l_chanel)
    #unir los canales de LAB 
    chanels_unidos = cv2.merge((cl,a_chanel,b_chanel))
    #regresar la imagen a BGR
    final_img = cv2.cvtColor(chanels_unidos,cv2.COLOR_LAB2BGR)
    res = np.hstack((img,final_img)) #stacking images side-by-side
    cv2.imwrite("Resultado.jpg",res)
    #cv2.imshow("res",res)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    skinmapFinal = cv2.morphologyEx(skinmap, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("skinmap.jpg",skinmapFinal)
    return skinmapFinal

def floodFill(image):
    print("floodFill")
    #original_img = image
    #cv2.imshow("antes del floodfill ",image)
    #se crea una imagen con threshold puesto a 220 y que ademas invierte la imagen
    th, imagen_threshold = cv2.threshold(image,220,255,cv2.THRESH_BINARY_INV)
    imagenInvertida = imagen_threshold.copy()
    #crear mascara para el flood fill se ocupa que sea 2 pixeles mayor que la original
    h, w = imagen_threshold.shape[:2]
    mask = np.zeros((h+2, w+2),np.uint8)
    #Floodfill desde el punto 0,0
    cv2.floodFill(imagenInvertida,mask,(0,0),255)
    #invertir floodfilled
    im_floodfill_inv = cv2.bitwise_not(imagenInvertida)
    #combinar las dos imagenes
    im_out = imagen_threshold | im_floodfill_inv
    kernel = np.ones((5,5),np.uint8)
    skinmapFinal = cv2.morphologyEx(imagenInvertida, cv2.MORPH_CLOSE, kernel)
    imgFinal = im_out & skinmapFinal
    image, contours, hierarchy = cv2.findContours(imgFinal,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("imagen con el closing",image)
    return contours,image

def getBiggestThreeContours(contours):
    """
        Esta funcion obtiene las 3 areas mas grandes dentro de los contornos
    """
    convexPoints = []
    areasMayores = []
    areas = []
    indices = []
    for n in contours:
        areas.append(cv2.contourArea(n))
    for area in range(3):
        a = np.amax(areas)
        index = areas.index(a)
        areas[index] = 0
        areasMayores.append(contours[index])
        indices.append(index)
    convexPoints.append(np.concatenate([areasMayores[0],areasMayores[1],areasMayores[2]]))
    return indices,convexPoints

def convexHull(image, convexPoints):
    print("Creando el ConvexHull...")
    hull = [cv2.convexHull(x) for x in convexPoints]
    final = cv2.drawContours(image,hull,-1,(100,255,100),5)
    #cv2.imshow('ALV2',final)
    return hull,final

def eliminateSmallAreas(image, contours):
    print("Eliminando Areas Insignificantes...")
    #cv2.imshow("ANTES DE BORRAR",image)
    #Regresar imagen a BGR para eliminar pixeles 
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    areas = [cv2.contourArea(x) for x in contours]
    print("Area minima valida:",(sum(areas)/100)*.2,"%")
    limit = (sum(areas)/100)*.2
    areasInvalidas = [x for x in areas if(x < limit)]
    cntListInvalidos = [c for c in contours if(cv2.contourArea(c) in areasInvalidas)]
    for cnt in cntListInvalidos:
        #puntos extremos de los contornos
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        for x in range(leftmost[0],rightmost[0]+1):
            for y in range(topmost[1],bottommost[1]+1):
                image.itemset((y,x,0),0)
                image.itemset((y,x,1),0)
                image.itemset((y,x,2),0)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image, contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("DespUEs DE BORRAR",image)
    return image, contours

def getHsvMean(contour,borders):
    print("Obteniendo la mediana del valor HSV...")
    #esta funcion debe obtener las medianas de los valores H, S y V que pertenecen
    #al area del contorno
    img = cv2.resize(ORIGINAL_IMG,(640,480),interpolation = cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h = [] 
    s = [] 
    v = []
    for x in range(borders[0][0],borders[1][0]+1):
        for y in range(borders[2][1],borders[3][1]+1):
            h.append(img[y,x][0])
            s.append(img[y,x][1])
            v.append(img[y,x][2])     
    return np.mean(h),np.mean(s),np.mean(v)

def getRectangularity(contour):
    print("Obteniendo La Rectangularidad del contorno...")
    #para obtener la rectangularidad de un contorno es igual a area De La Figura / Area del bounding rectangle
    areaFig = cv2.contourArea(contour)
    #obtener el area del recntagulo 
    rect = cv2.minAreaRect(contour)
    try:
        return areaFig/(rect[1][0]*rect[1][1])
    except:
        return 0

def getMeanGray(skinMap, contour, borders):
    print("Obteniendo La Mediana de Gris")
    #regresar la mediana del valor de gris en el contorno
    n = 0
    grayValue = 0
    for x in range(borders[0][0],borders[1][0]+1):
        for y in range(borders[2][1],borders[3][1]+1):
            n += 1
            if(skinMap[y,x] == 255):
                grayValue += 1
    return grayValue / n

def getLongitudRectangulo(contour):
    izq = tuple(contour[contour[:,:,0].argmin()][0])
    der = tuple(contour[contour[:,:,0].argmax()][0])
    arriba = tuple(contour[contour[:,:,1].argmin()][0])
    abajo =tuple(contour[contour[:,:,1].argmax()][0])
    return ((((der[1]-izq[1])**2)+((arriba[0]-abajo[0])**2))**(1/2))

def getEccentricity(contour):
    print("Obteniendo La Excentrecidad del contorno...")
    #regresar el valor de excentiricidad
    #en un "minimum bounding box " la excentricidad
    #puede obtenerse dividiendo la longitud del rectangulo por su ancho
    #obtener el area del recntagulo 
    rect = cv2.minAreaRect(contour)
    longitudAxisMayor = getLongitudRectangulo(contour)
    return longitudAxisMayor/rect[1][0]

def getEllipcity(contour):
    print("Obteniendo La Elipcidad Del Contorno...")
    #regresar el valor de elipcidad
    #la elipcidad esta dada por 1 - axias menor sobre el axis mayor
    try:
        e = cv2.fitEllipse(contour)
    except:
        print("OH NOOOOOOOO HUBO UN ERROR!!!")
        return 0
    return 1-(e[1][0]/e[1][1])

def getOrientation(contour):
    print("Obteniendo La Orientacion del contorno...")
    #regresar el valor de orientacion
    izq = tuple(contour[contour[:,:,0].argmin()][0])
    der = tuple(contour[contour[:,:,0].argmax()][0])
    arriba = tuple(contour[contour[:,:,1].argmin()][0])
    abajo =tuple(contour[contour[:,:,1].argmax()][0])
    return np.arctan((der[1]-izq[1])/(abajo[0]-arriba[0]))

def getCantidadPixelesEnBordes(skinmap,contour,borders):
    print("Obteniendo La Cantidad De Piexeles Que Tocan Los Bordes Del Contorno...")
    #regresar el numero de pixeles de la region de piel que tocan los bordes
    #de la imagen completa
    touching = False
    pixels = 0
    #obtenemos las esquinas del contorno
    if(borders[0][1] == 0):
        touching = True
    elif(borders[0][1] == 640):
        touching = True
    elif(borders[2][0] == 480):
        touching = True
    elif(borders[3][0] == 0):
        touching = True
    if(touching == True):
        for x in range(borders[0][0],borders[1][0]+1):
            for y in range(borders[2][1],borders[3][1]+1):
                if(skinmap[y,x] == 255):
                    pixels += 1
        return pixels
    return 0

def getCantidadDeEsquinasTocadas(contour,borders):
    print("Obteniendo La Cantidad De Esquinas Que Se Tocan En El Contorno...")
    #regresa la cantidad de esquinas en la imagen tocadas por la region de piel
    touching = 0
    #obtenemos las esquinas del contorno
    #tolerancia de 3 pixeles
    if(borders[0][1] <= 3):
        touching +=1
    elif(borders[0][1] >= 637):
        touching += 1
    elif(borders[2][0] >= 477):
        touching += 1
    elif(borders[3][0] <= 3):
        touching += 1
    return touching

def calculateTotalSkin(skinmap):
    print("Calculando La Cantidad De Piel Total...")
    global TOTAL_SKIN
    TOTAL_SKIN = 0
    for x in range(640):
        for y in range(480):
            if(skinmap[y,x] == 255 or skinmap[y,x] == 190):
                TOTAL_SKIN += 1

def getSkinAmount(skinmap, contour, borders):
    print("Obteniendo La Cantidad De Piel Que Se Encuentra Dentro Del Contorno...")
    #regresa el porcentaje de piel comparada con la imagen completa
    areaSkin = 0
    for x in range(borders[0][0],borders[1][0]+1):
        for y in range(borders[2][1],borders[3][1]+1):
            if(skinmap[y,x] == 255):
                areaSkin += 1
    return (areaSkin/TOTAL_SKIN)*100
    
def getHueStandardDeviation(contour,borders):
    print("Obteniendo La Desviacion Estandar Del Componente HSV Del Contorno...")
    #regresa la desviacion estandar del componente de HSV
    imgHSV = cv2.resize(ORIGINAL_IMG,(640,480),interpolation = cv2.INTER_CUBIC)
    imgHSV = cv2.cvtColor(imgHSV,cv2.COLOR_BGR2HSV)
    hue = []
    for x in range(borders[0][0],borders[1][0]+1):
        for y in range(borders[2][1],borders[3][1]+1):
            hue.append(imgHSV[y,x][0])
    hueMean = np.mean(hue)
    desviacionSTD = list(map(lambda x: (x - hueMean)**2,hue))
    desviacionSTD = math.sqrt(sum(desviacionSTD)/len(hue))
    return desviacionSTD

def getPerimeter(contour):
    print("Obteniendo El Perimetro Del Contorno...")
    #regresa el perimetro
    perimetro = 0
    for indice in range(len(contour)-1):
        perimetro += (((contour[indice+1][0][1]-contour[indice][0][1])**2)+((contour[indice+1][0][0]-contour[indice][0][0])**2))**(1/2)
    perimetro += (((contour[-1][0][1]-contour[0][0][1])**2)+((contour[-1][0][0]-contour[0][0][0])**2))**(1/2) 
    return perimetro

def getCompactness(contour):
    print("Obteniendo La Relacion De Compacticidad Del Contorno...")
    #regresa la relacion entre la region de piel y su perimetro
    return (4*math.pi*cv2.contourArea(contour))/(cv2.arcLength(contour,True)**2)

def getCentroid(contour):
    print("Obteniendo El Valor X y Y Del Centroide Del Contorno...")
    #regresa el centroide del area de la piel
    M = cv2.moments(contour)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except:
        cx = 0
        cy = 0
    return cx, cy

def extractFeatures(skinMap,skinArea):
    features = {"Rectangularity":0,"HMean":0,"SMean":0,"Vmean":0,"MeanGray":0,"Eccentricity":0,"Ellipcity":0,"Orientation":0,"Amount of touching pixels":0
    ,"Amount of Skin":0,"Number of touched corners":0,"Hue std":0,"Perimeter":0,"Compactness":0,"CentroideX":0,"CentroideY":0}
    borders = []
    borders.append(tuple(skinArea[skinArea[:,:,0].argmin()][0]))
    borders.append(tuple(skinArea[skinArea[:,:,0].argmax()][0]))
    borders.append(tuple(skinArea[skinArea[:,:,1].argmin()][0]))
    borders.append(tuple(skinArea[skinArea[:,:,1].argmax()][0]))
    features["Rectangularity"] = getRectangularity(skinArea)
    h,s,v = getHsvMean(skinArea,borders)
    features["HMean"] = h
    features["SMean"] = s
    features["Vmean"] = v 
    features["MeanGray"] = getMeanGray(skinMap,skinArea,borders)
    features["Ellipcity"] = getEllipcity(skinArea)
    features["Orientation"] = getOrientation(skinArea)
    features["Amount of touching pixels"] = getCantidadPixelesEnBordes(skinMap,skinArea,borders)
    features["Amount of Skin"] = getSkinAmount(skinMap,skinArea,borders)
    features["Number of touched corners"] = getCantidadDeEsquinasTocadas(skinArea,borders)
    features["Hue std"] = getHueStandardDeviation(skinArea,borders)
    features["Perimeter"] = getPerimeter(skinArea)
    features["Compactness"] = getCompactness(skinArea)
    x, y = getCentroid(skinArea)
    features["CentroideX"] = x
    features["CentroideY"] = y
    features["Eccentricity"] =  getEccentricity(skinArea)
    return features 

def checkSpatial(skinmap):
    print("Checkspatial")
    skinmap = cv2.cvtColor(skinmap, cv2.COLOR_GRAY2BGR)
    box = 480 / 9
    skin = 0
    totalSkinArea = 0
    for x in range(640):
        for y in range(int(box*4),int(box*5)+1):
            totalSkinArea += 1 
            if(skinmap[y,x][0] == 255):
                skin += 1
    if(((skin/totalSkinArea)*100) < 29):
        return False
    else:
        return True

def checkFace(skinmap):
    print("Detectando La Cara En La Imagen...")
    try:
        skin = 0
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        image = cv2.resize(ORIGINAL_IMG,(640,480),interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,100,150),2)
        for i in range(x,x+w):
            for j in range(y,y+h):
                if(skinmap[j,i] == 255):
                    skin += 1
                image.itemset((j,i,0),255)             
        #cv2.imshow("deteccion de face",image)
        #cv2.waitKey()
        return (skin/TOTAL_SKIN)*100
    except:
        print("No Se Detecto Una Cara...")
        return 0

def eraseArea(image,skinArea):
    print("Borrando Areas Invalidas...")
    izq = tuple(skinArea[skinArea[:,:,0].argmin()][0])
    der = tuple(skinArea[skinArea[:,:,0].argmax()][0])
    arriba = tuple(skinArea[skinArea[:,:,1].argmin()][0])
    abajo = tuple(skinArea[skinArea[:,:,1].argmax()][0])
    for x in range(izq[0],der[0]+1):
        for y in range(arriba[1],abajo[1]+1):
            image.itemset((y,x,0),0)
            image.itemset((y,x,1),0)
            image.itemset((y,x,2),0)
    
    return image

def shapeElimination(img,skinAreas,features):
    print("Eliminacion De Formas Invalidas...")
    longitudImagen = 640
    areaImagen = (480*640)
    #cv2.imshow("antes de shape elimination",img)
    for indice in range(len(skinAreas)):
        rect = cv2.minAreaRect(skinAreas[indice])
        if(features[indice]["Rectangularity"] > 0.75 and features[indice]["Compactness"] > 0.75 and features[indice]["Ellipcity"] > 0.75):
            img = eraseArea(img,skinAreas[indice])
            print("Condicion 1: eliminado por ser demasiado compacto.")
        elif(features[indice]["Compactness"] < 0.01):
            img = eraseArea(img,skinAreas[indice])
            print("Condicion 2: eliminado por no ser lo suficientemente compacto.")
        elif(rect[1][0]*1.1 > longitudImagen and cv2.contourArea(skinAreas[indice])*2 < areaImagen and features[indice]["Rectangularity"] > 0.60):
            img = eraseArea(img,skinAreas[indice])
            print("Eliminar por prevencion de atardecer, por posible confusion por causa del color.")
    #cv2.imshow("despues de shape elimination",img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return img,contours

class Node():
    def __init__(self,pos):
        self.posx = pos[1]
        self.posy = pos[0]
        self.pos = pos

def generaHijos(nodo,imagen,limitesMIN,limitesMAX,visitados):
    hijos = []
    noPiel = 0
    piel = 0
    #up
    if(nodo.posy + 1 < limitesMAX[1] and nodo.posx > limitesMIN[0] and nodo.posx < limitesMAX[0] and (nodo.posy+1,nodo.posx) not in visitados):
        if(imagen[nodo.posy+1,nodo.posx][0] != 100):
            if(imagen[nodo.posy+1,nodo.posx][0] == 255):
                piel += 1
            else:
                noPiel += 1
            node = Node((nodo.posy + 1,nodo.posx))
            hijos.append(node)
            visitados.add(node.pos)
   #left
    if(nodo.posx-1 > limitesMIN[0] and nodo.posy > limitesMIN[1] and nodo.posy < limitesMAX[1] and (nodo.posy,nodo.posx-1) not in visitados):
        if(imagen[nodo.posy,nodo.posx-1][0] != 100):
            if(imagen[nodo.posy,nodo.posx-1][0] == 255):
                piel += 1
            else:
                noPiel += 1
            node = Node((nodo.posy,nodo.posx-1))
            hijos.append(node)
            visitados.add(node.pos)
    #right
    if(nodo.posx + 1 < limitesMAX[0] and nodo.posy > limitesMIN[1] and nodo.posy < limitesMAX[1] and (nodo.posy,nodo.posx+1) not in visitados):
        if(imagen[nodo.posy,nodo.posx+1][0] != 100):
            if(imagen[nodo.posy,nodo.posx+1][0] == 255):
                piel += 1
            else:
                noPiel += 1
            node = Node((nodo.posy,nodo.posx+1))
            hijos.append(node)
            visitados.add(node.pos)
   #down
    if(nodo.posy-1 > limitesMIN[1] and nodo.posx > limitesMIN[0] and nodo.posx < limitesMAX[0] and (nodo.posy-1,nodo.posx) not in visitados):
        if(imagen[nodo.posy-1,nodo.posx][0] != 100):
            if(imagen[nodo.posy-1,nodo.posx][0] == 255):
                piel += 1
            else:
                noPiel += 1
            node = Node((nodo.posy-1,nodo.posx))
            hijos.append(node)
            visitados.add(node.pos)
    return hijos,imagen,noPiel,piel

def bfs(imagen,minX,minY,maxX,maxY):
    piel = noPiel = 0
    inicio = ((int((maxY-minY)/2)+minY,int((maxX-minX)/2)+minX))
    objetivo = (maxX,maxY)
    frontera = []
    visitados = set()
    nodo = Node(inicio)
    frontera.append(nodo)
    visitados.add(inicio)
    while(len(frontera)>0):
        nodo = frontera.pop(0)
        if(objetivo[1] == nodo.posy and objetivo[0] == nodo.posx):
            break
        childs, imagen, noskin, skin = generaHijos(nodo,imagen,(minX,minY),objetivo,visitados)
        piel += skin
        noPiel += noskin
        for n in childs:
            frontera.append(n)
    return imagen, piel, noPiel

def getConvexHullFillrate(image,skinArea,oldContours):
    print("Obteniendo La Cantidad De Piel Dentro Del Convex Hull...")
    izq = tuple(skinArea[skinArea[:,:,0].argmin()][0])
    der = tuple(skinArea[skinArea[:,:,0].argmax()][0])
    arriba = tuple(skinArea[skinArea[:,:,1].argmin()][0])
    abajo = tuple(skinArea[skinArea[:,:,1].argmax()][0])
    image,piel,noPiel = bfs(image,izq[0],arriba[1],der[0],abajo[1])
    for cnt in oldContours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        cv2.drawContours(image,[box],0,(150,50,0),2)
    #cv2.imshow("fillrate",image)
    print("PIEL: ",piel," NO PIEL: ",noPiel)
    print("%PIEL en el convex hull: ",(piel/(piel+noPiel)*100))
    return (piel/(piel+noPiel)*100), image

def fillFeatures(filename,label,indices,features,params = []):
    ImageFeatures = {"Filename":"","Clasificacion":"","Rectangularity1":0,"Rectangularity2":0,"HMean":0,"SMean":0,"Vmean":0,"MeanGray":0,"Eccentricity":0,
    "Ellipcity":0,"Orientation":0,"Amount of touching pixels":0,"Amount of Skin":0,"Number of touched corners":0,
    "Hue std":0,"Perimeter":0,"Compactness1":0,"CentroideX":0,"CentroideY":0,"Size1":0,"Size2":0,"Polygonfillrate":0,"Skinfillrate":0,"Areacara":0}
    ImageFeatures["Filename"] = filename
    ImageFeatures["Rectangularity1"] = features[indices[0]]["Rectangularity"]
    ImageFeatures["HMean"] = features[indices[0]]["HMean"]
    ImageFeatures["SMean"] = features[indices[0]]["SMean"]
    ImageFeatures["Vmean"] = features[indices[0]]["Vmean"]
    ImageFeatures["MeanGray"] = features[indices[0]]["MeanGray"]
    ImageFeatures["Eccentricity"] = features[indices[0]]["Eccentricity"]
    ImageFeatures["Ellipcity"] = features[indices[0]]["Ellipcity"]
    ImageFeatures["Orientation"] = features[indices[0]]["Orientation"]
    ImageFeatures["Amount of touching pixels"] = features[indices[0]]["Amount of touching pixels"]
    ImageFeatures["Amount of Skin"] = features[indices[0]]["Amount of Skin"]
    ImageFeatures["Number of touched corners"] = features[indices[0]]["Number of touched corners"]
    ImageFeatures["Hue std"] = features[indices[0]]["Hue std"]
    ImageFeatures["Perimeter"] = features[indices[0]]["Perimeter"]
    ImageFeatures["Compactness1"] = features[indices[0]]["Compactness"]
    ImageFeatures["CentroideX"] = features[indices[0]]["CentroideX"]
    ImageFeatures["CentroideY"] = features[indices[0]]["CentroideY"]
    ImageFeatures["Size1"] = params[1]
    ImageFeatures["Size2"] = params[2]
    ImageFeatures["Polygonfillrate"] = params[0]
    ImageFeatures["Areacara"] = params[3]
    ImageFeatures["Rectangularity2"] = features[indices[1]]["Rectangularity"]
    ImageFeatures["Skinfillrate"] = (TOTAL_SKIN/(640*480))*100
    ImageFeatures["Clasificacion"] = label
    return ImageFeatures

def saveImageFeatures(feature,dataset):
    mode = ""
    if(dataset == "Caracteristicas.csv"):
        mode = "a+"
    elif(dataset == "Sample.csv"):
        mode = "w+"
        
    file_exists = os.path.isfile(dataset)
    with open(dataset, mode) as csvfile:
        fieldnames = ['Filename','Clasificacion', 'Rectangularity1','Rectangularity2','HMean','SMean','Vmean','MeanGray','Eccentricity',
        'Ellipcity','Orientation','Amount of touching pixels','Amount of Skin','Number of touched corners','Hue std','Perimeter'
        ,'Compactness1','CentroideX','CentroideY','Size1','Size2','Polygonfillrate','Skinfillrate','Areacara']
        
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=fieldnames)
        if(not file_exists or mode == "w+"):
            writer.writeheader()
        writer.writerow(feature)

def classifySkinMap(skinmap,filename,dataset,label):
    areaCara = 0
    contours, image = floodFill(skinmap)
    image,newContours = eliminateSmallAreas(image, contours)
    features = []
    calculateTotalSkin(image)
    for cnt in newContours:
        features.append(extractFeatures(image, cnt))
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img, newContours = shapeElimination(img,newContours,features)
    indices, convexPoints = getBiggestThreeContours(newContours)
    if(checkSpatial(img) == False):
        print("Probablemente No Es Pornografica...")
    else:
        print("Probablemente Es Pornografica!!!")
        areaCara = checkFace(img)
    if(areaCara > 38):
        print("Se Detecto Una Cara...")
    else:
        print("No Se Detecto Una Cara...")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hull,img = convexHull(image, convexPoints)
    pollygonFillrate,img = getConvexHullFillrate(img,hull[0],newContours)
    try:
        sizeOne = (cv2.contourArea(newContours[indices[0]])/(640*480))*100
        sizeTwo = (cv2.contourArea(newContours[indices[1]])/(640*480))*100
        ImageFeatures = fillFeatures(filename,label,indices,features,params=[pollygonFillrate,sizeOne,sizeTwo,areaCara])
    except:
        try:
            sizeOne = (cv2.contourArea(newContours[indices[0]])/(640*480))*100
            ImageFeatures = fillFeatures(filename,label,indices,features,params=[pollygonFillrate,sizeOne,0,areaCara])
        except:
            ImageFeatures = fillFeatures(filename,label,indices,features,params=[pollygonFillrate,0,0,areaCara])     
    saveImageFeatures(ImageFeatures,dataset)
    return img

def SVMEffectivity():
    df = pd.read_csv("Caracteristicas.csv")
    labels = np.asarray(df.Clasificacion)
    le = LabelEncoder()
    le.fit(labels)
    # apply encoding to labels
    labels = le.transform(labels)
    df_selected = df.drop(['Filename', 'Clasificacion',"HMean", "SMean","Vmean","MeanGray","Eccentricity","Ellipcity","Orientation","Hue std","CentroideY","Perimeter",
    "CentroideX","Areacara","Amount of touching pixels","Number of touched corners"], axis=1)

    df_f = df_selected.to_dict(orient='records')

    vec = DictVectorizer()
    features = vec.fit_transform(df_f).toarray()
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, 
        test_size=0.25, random_state=42)

    #clf = RandomForestClassifier()
    clf = SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.00001, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    clf.fit(features_train,labels_train)
    acc_test = clf.score(features_test,labels_test)
    print("\n\n\n\nTEST ACCURACY WITH SVC(Support Vector Classifier):",acc_test)
    
    clf = RandomForestClassifier()
    clf.fit(features_train,labels_train)
    acc_test = clf.score(features_test,labels_test)
    print("TEST ACCURACY WITH Random Forest Classifier:",acc_test)
    
    clf = GaussianNB()
    clf.fit(features_train,labels_train)
    acc_test = clf.score(features_test,labels_test)
    print("TEST ACCURACY WITH Naive Bayes:",acc_test,"\n\n\n\n")

def testNewSample():
    """
        Por falta de tiempo esta funcion sera implementada completamente despues.
        hay un problema con las etiquetas.
    """
    """global ORIGINAL_IMG 
    reading = True
    dataset = "Sample.csv"
    while(reading == True):
        print("\n\nPresiona S para salir")
        print("\n\nEscribe El Nombre Completo De La Imagen Junto A La Extension .jpg\n\n")
        imagen = str(input())
        if(imagen == "s" or imagen == "S"):
            opc = 0
            return
        else:
            try:
                ORIGINAL_IMG = cv2.imread(imagen)
                skinmap = skinDetection(imagen)
                skinmap = cv2.cvtColor(skinmap,cv2.COLOR_BGR2GRAY)
                classifySkinMap(skinmap,imagen,dataset,"NO SE")
                reading = False
            except:
                print("\n\n\nHubo Un Error Con La Imagen Revisa Que Exista En El Mismo Directorio Donde Se Encuentre El Script y Que No Este En Escala De Grises!!!\n\n\n")
    """
    df = pd.read_csv("Caracteristicas.csv")
    df_sample = pd.read_csv("CaracteristicasTEST.csv")
    
    labels = np.asarray(df.Clasificacion)
    labelsTest = np.asarray(df_sample.Clasificacion)
    le = LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    labels_test = le.transform(labelsTest)

    #df_selected = df.drop(['Filename', 'Clasificacion'], axis=1)
    #df_sample = df_sample.drop(['Filename', 'Clasificacion'], axis=1)
    df_selected = df.drop(['Filename', 'Clasificacion',"HMean", "SMean","Vmean","MeanGray","Eccentricity","Ellipcity","Orientation","Hue std","CentroideY","Perimeter",
    "CentroideX","Areacara","Amount of touching pixels","Number of touched corners"], axis=1)
    
    df_sample_selected = df_sample.drop(['Filename', 'Clasificacion',"HMean", "SMean","Vmean","MeanGray","Eccentricity","Ellipcity","Orientation","Hue std","CentroideY","Perimeter",
    "CentroideX","Areacara","Amount of touching pixels","Number of touched corners"], axis=1)
    
    df_f = df_selected.to_dict(orient='records')  
    df_test = df_sample_selected.to_dict(orient='records')
    
    vec = DictVectorizer()
    vec.fit(df_test)
    features = vec.transform(df_f).toarray()
    testFeatures = vec.transform(df_test).toarray()
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, 
        test_size=0.25, random_state=42)

    clf = GaussianNB()
    """clf = SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.00001, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)"""
    #clf = RandomForestClassifier()
    #clf = RandomForestClassifier(n_estimators = 500, random_state = 50)
    clf.fit(features_train,labels_train)
    
    print("Resultados de la prediccion de las nuevas imagenes")
    print(clf.predict(testFeatures))


    print("Resultados de la prediccion del test")
    print(clf.predict(features_test))

    l_predict = clf.predict(features_test)
    print("\n\nEfectividad del modelo = ",accuracy_score(labels_test, l_predict))

    print("\n\nMatriz de confusion\n",pd.DataFrame(
    confusion_matrix(labels_test, l_predict),
    columns=['Predicted Pornograficas', 'Predicted NoPornograficas'],
    index=['True Pornograficas', 'True NoPornograficas']
    ))
    exit()

def main():
    global ORIGINAL_IMG 
    opc = 0
    while(opc==0):
        try:
            print("\n\n\n\n\n\n\n*****Detector De Pornografia En Imagenes 9000*****\n\n")
            print("Elige La Opcion Que Desees Ejecutar:\n\n1.-Ingresar Imagen Para Incrementar Tamaño DataSet")
            print("2.-Ver Efectividad Del Modelo")
            print("3.-Clasificar Una Nueva Imagen(AUN NO FUNCIONA COMPLETAMENTE)\n\n5.-Para Salir...")
            opc = int(input())
        except:
            print("Error Con La Opcion Seleccionada!!")
            opc = 0
        while(opc > 0 and opc < 5):
            if(opc == 1):
                print("\n\nPresiona S para salir")
                print("\n\nEscribe El Nombre Completo De La Imagen Junto A La Extension .jpg\n\n")
                imagen = str(input())
                if(imagen == "s" or imagen == "S"):
                    opc = 0
                else:
                    while(1):
                        print("Escribe La Etiqueta De La Imagen ej. NoPornografica o Pornografica\n\n")
                        label = str(input())
                        if(label == "NoPornografica"):
                            break
                        elif(label == "Pornografica"):
                            break 
                    try:
                        ORIGINAL_IMG = cv2.imread(imagen)
                        skinmap = skinDetection(imagen)
                        skinmap = cv2.cvtColor(skinmap,cv2.COLOR_BGR2GRAY)
                        classifySkinMap(skinmap,imagen,"Caracteristicas.csv",label)
                    except:
                        print("\n\n\nHubo Un Error Con La Imagen Revisa Que Exista En El Mismo Directorio Donde Se Encuentre El Script y Que No Este En Escala De Grises!!!\n\n\n")
            elif(opc == 2):
                SVMEffectivity()
                opc = 0
            elif(opc == 3):
                testNewSample()
                opc = 0
            elif(opc == 5):
                exit()
    
if(__name__=="__main__"):
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    main()