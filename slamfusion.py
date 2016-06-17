#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
SLAMFusion
1-Aplicar outros metodos aos ros bags
2-identificar rosbag com ground truths sem metodos slam.
3-Mapas produzidos por cada um dos metodos para os datasets.
4-Preceber o conteudo dos rosbags.
5-arranjar forma de colocar os mapas do mesmo tamanho, dinamicamente.

Ver state of the art no slam
Revistas onde estao a aparecer os metodos slam.
99-Estudar como ter os mapas dinamicamente num nodo ros.




20-4-2016
NOTA IMP: VX-> Anteriores MKX-> REcentes
Ate Maio.


1-report de cloud robotics. Procurar no ieee xplore em 2016 algum review ou mais papers que falem sobre isto

1-Estado da arte
O que é um algoritmo cloud robotic (o que se considera,) O ros é cloud robotic?

como fazer a nossa propria cloud? o ros faz? 

ver frameworks para cloud robotics.



2-Acabar o codigo (este especifico aberto)
Ver como passar o opencv para gpu


3- Paper para revista com tudo isto.




#####

TODO list:

Fazer alinhamento dos 3 mapas em relação a ground truth. (Oposto do que esta feito)
 ++DONE++Colocar todos os mapas com as mesmas dimensões
  
Definir o maior mapa pelo size, A= maior B= Menor. guardar informação no tuplo
(função process)

- mudar floats32 para int32
- experiencias com mapas novos, ver melhorias de codigo
- Ver threads aplicadas ao pyopencl

Método hierárquico aplicado a procura de mapas
(Scale do mapa 100x100 para 10x10)
Ver no opencv: Correcção das rotações, acrescentar -1 em vez de 0




TODO Done list:

- Correr codigo com placa nova.
- ver salto de tempo entre ROTMAX = 5(3 secs) e ROTMAX = 45(6 secs) ROTMAX = 90 (10 secs)
- Tratar de mudar a grafica principal

Bugs to fix
############


Fixed Bugs
################

Pesquisa apenas procurava para traz entre (-10->0), agora procura (-10->0->10)

Additions
#############

ArrayHalfMaker (start)

Global Variables
###################
- ROTMAX = 5 ->Max angle for rotations
- countkarto = 0 -> Karto Frame counter
- datakarto=OccupancyGrid() -> Karto Map Information
- countgmapping = 0 -> Gmapping Frame counter
- datagmapping=OccupancyGrid() -> Gmapping Map Information
- countrtab = 0 -> RTABMap Frame counter
- datartab=OccupancyGrid() -> RTABMap Map Information
- countSLAMFusion = 0 -> SLAMFusion Frame counter
- dataSLAMFusion = OccupancyGrid() -> Slamfusion Map Information

Methods
########

"""

from __future__ import absolute_import, print_function
import rospy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
import Image
import os
import numpy as np
import thread
import threading
import itertools
import sys
import operator
import pyopencl as cl
import cv2
import time

from math import sqrt


ROTMAX = 20
countkarto = 0
datakarto=OccupancyGrid()
countgmapping = 0
datagmapping=OccupancyGrid()
countrtab = 0
datartab=OccupancyGrid()
countSLAMFusion = 0
dataSLAMFusion = OccupancyGrid()
def findmaxcoord(array):
	sizeX = []
	sizeY = []
	
	for i in array:
		sizeX.append(i[0])
		sizeY.append(i[1])
	sizeX.sort(reverse=True)
	sizeY.sort(reverse=True)
	return(sizeX[0]+100,sizeY[0]+100)
	
def arrayHalfMaker(start):
	"""Recieves a as a parameter a start value, divides that value by 2 until its equal to 1, appending the results to a array.
	
	:param start: Start value
	:type start: Integer
	:returns: Integer Array -- the various results calculated.
	
	"""
	aux = start
	array = [aux]
	while (aux != 1):
		aux=aux/2
		array.append(aux)
	return array

def mkimg(array,name):
	"""Creates a image from the array information
	
	:param array: Map/Data array with integers that range from -1 to 1.
	:type array: Numpy float32 array.
	:param name: name for the resulting file.
	:type name:  String
	
	"""
	aux=np.array(array)
	for i in np.nditer(aux,op_flags=['readwrite']):
		if(i==0):
			i[...]=127
		elif(i<0):
			i[...]=240
		elif(i>0):
			i[...]=0
	new_img = Image.new("L", (aux.shape[1],aux.shape[0]), "white")
	listimg = aux.flatten()
	new_img.putdata(listimg)
	new_img.save(name)
	
	"""
	Old Values
		if(i==0):
			i[...]=55
		elif(i<0):
			i[...]=35
		elif(i>0):
			i[...]=100
	"""
	
def treatimg(array):
	array = array[:,:,0]
	aux=np.array(array).astype(np.float32)
	np.savetxt('./treatimg_out.txt',aux,fmt='%.0f')
	for i in np.nditer(aux,op_flags=['readwrite']):
		if(i>127):
			i[...]=-1
		elif(i<127):
			
			i[...]=1
			
		else:
			i[...]=0
		
	return aux
	
def karto(data):
	"""Karto node subscriber method. 
	Sends the recieved karto node information to treatment method.
	
	:param data: OccupancyGrid Data, containing map information and metadata like height, width or resulution of the map.
	:type data: OccupancyGrid struct.
	
	"""
	
	global countkarto
	countkarto=countkarto+1
	print ("Karto->",countkarto)
	treatment(data,'karto',countkarto)

def gmapping(data):
	"""Gmapping node subscriber method. 
	Sends the recieved karto node information to treatment method.
	
	:param data: OccupancyGrid Data, containing map information and metadata like height, width or resulution of the map.
	:type data: OccupancyGrid struct.
	
	"""
	global countgmapping
	countgmapping = countgmapping+1
	print ("Gmapping->",countgmapping)
	treatment(data,'gmapping',countgmapping)
    
def rtabmap(data):
	"""Rtabmap node subscriber method. 
	Sends the recieved karto node information to treatment method.
	
	:param data: OccupancyGrid Data, containing map information and metadata like height, width or resulution of the map.
	:type data: OccupancyGrid struct.
	
	"""
	global countrtab
	countrtab=countrtab+1
	print ("Rtab->",countrtab)
	treatment(data,'rtabmap',countrtab)

def defmap(data):
    print ("map->",0)
    treatment(data,'originalmap',0)
    
def matrixmaker(data):
	"""Creates a map matrix used throughtout the program from a integer list.
	
	:param data: OccupancyGrid Data, containing map information and metadata like height, width or resulution of the map.
	:type data: OccupancyGrid struct.
	:returns: Integer Matrix -- A 2D matrix with height and width of the map, with points ranging from -1 to 1.
	
	"""
	matriz = np.array(data.data).reshape(data.info.height,data.info.width)
	matrizCorr = np.flipud(matriz)
	for i in np.nditer(matrizCorr,op_flags=['readwrite']):
		if(i==100):
			i[...]=1
	return matrizCorr
	
def treatment(data,name,count):
	"""Creates various ways to show the obtained informations.
	
	- Text mode.
	- Image mode.
	
	:param data: OccupancyGrid Data, containing map information and metadata like height, width or resulution of the map.
	:type data: OccupancyGrid struct.
	:returns: Integer Matrix -- A 2D matrix with height and width of the map, with points ranging from -1 to 1.
	
	"""
	if not os.path.exists('./'+name):
		os.makedirs('./'+name)
	imglist = matrixmaker(data)
	imglist.dump('./'+name+'/'+name+str(count)+'MATRIZ.bin')
	np.savetxt('./'+name+'/'+name+str(count)+'.txt',imglist,fmt='%.0f')
	mkimg(imglist,'./'+name+'/'+name+str(count)+'.tiff')
	j=0
	f1=open('./'+name+'/'+name+str(count)+'RAW.txt','w+')
	for i in data.data:
		if(i==0):
			f1.write(' ')
		if(i==1):
			f1.write('X')
		if(i==-1):
			f1.write('1')
		j=j+1
		if(j==data.info.width):
			f1.write('\n')
			j=0
	f3=open('./'+name+'/'+name+str(count)+'info.txt','w+')
	f3.write(str(data.info))
	upmap(data,name)
	
def listner():
	"""Starts the various ROS subscribers and the publisher.
	The subscribers are:
	
	- /Karto/map.
	- /Gmapping/map.
	- /RTABMap/map.
	
	The publisher is /SLAMfusion/map
	
	"""
	
	rospy.init_node('SLAMFusion',anonymous=False)
	rospy.Subscriber("slamkarto/map",OccupancyGrid,karto)
	rospy.Subscriber("gmapping/map",OccupancyGrid,gmapping)
	rospy.Subscriber("rtabmap/map",OccupancyGrid,rtabmap)
	rospy.Subscriber("/map",OccupancyGrid,defmap)
	rospy.Publisher('/SLAMFusion/map',OccupancyGrid, queue_size=10)
	rospy.spin()
    
def upmap(newmap,algorithm):
	""" Updates map informations then flows into calcmap method if there are maps for all methods.
	
	:param newmap: OccupancyGrid Data, containing map information and metadata like height, width or resulution of the map.
	:type newmap: OccupancyGrid struct.
	:param algorithm: OccupancyGrid Data, containing map information and metadata like height, width or resulution of the map.
	:type algorithm: String.
	
	
	"""
	
	print ('Upmap, yay!',algorithm)
	global datakarto
	global datagmapping
	global datartab
	if(algorithm == 'karto'):
		if(newmap!=datakarto):
			datakarto=newmap
	elif(algorithm == 'gmapping'):
		if(newmap!=datagmapping):
			datagmapping=newmap
	elif(algorithm == 'rtabmap'):
		if(newmap!=datartab):
			datartab = newmap
	if len(datagmapping.data) and len(datakarto.data) and len(datartab.data):
		calcmap()
		
def mkcrop(matrix,secshape,errortuple):
	rows,cols = secshape
	rotavar= errortuple[3]
	A = cv2.getRotationMatrix2D((cols/2,rows/2),rotavar,1)
	rot = cv2.warpAffine(matrix,A,(cols,rows),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS,borderValue=(-1, -1, -1, -1))
	return np.array(rot).astype(np.float32)
	#return np.array(rot[errortuple[1]:secshape[0]+errortuple[1],errortuple[2]:secshape[1]+errortuple[2]]).astype(np.float32)

def removeminusone(matrix):
	valtoremX = []
	valtoremY = []
	aux=0

	for i in range(0,matrix.shape[0]):
		for j in matrix[i]:
			if(j != -1.0):
				aux=aux+1
				break;		
		if(aux==0):
			valtoremX.append(i)
		else:
			aux=0
	matrix=np.delete(matrix,valtoremX,axis=0)
	matrix=np.rot90(matrix).astype(np.float32)
	
	for i in range(0,matrix.shape[0]):
		for j in matrix[i]:
			if(j != -1.0):
				aux=aux+1
				break;		
		if(aux==0):
			valtoremY.append(i)
		else:
			aux=0
	matrix=np.delete(matrix,valtoremY,axis=0)
	matrix=np.rot90(matrix).astype(np.float32)
	matrix=np.rot90(matrix).astype(np.float32)
	matrix=np.rot90(matrix).astype(np.float32)
	print("ValtoremX:",len(valtoremX))
	print("ValtoremY:",len(valtoremY))
	
	return matrix

def forceToSize(matrix,scaleb):
	while(matrix.shape[0] !=scaleb[0] ):
		#print('0.0',matrix.shape[0],scaleb[0])
		matrix = np.insert(matrix,0,-1,axis=0)
		if(matrix.shape[0] !=scaleb[0] ):
			#print('0.1',matrix.shape[0],scaleb[0])	
			matrix = np.insert(matrix,matrix.shape[0],-1,axis=0)

		
	while(matrix.shape[1] !=scaleb[1] ):
		#print('1.0',matrix.shape[1],scaleb[1])
		matrix = np.insert(matrix,0,-1,axis=1)
		if(matrix.shape[1] !=scaleb[1] ):
			#print('1.1',matrix.shape[1],scaleb[1])
			matrix = np.insert(matrix,matrix.shape[1],-1,axis=1)
	return matrix

def createnewmap(matrix,newshape,errortuple):
	new = np.ones(newshape)
	new = new*-1	
	for x in range(0,matrix.shape[0]):
		for y in range(0,matrix.shape[1]):
			new[x+errortuple[1]][y+errortuple[2]] = matrix[x][y]
	return new


def calcmap():
	""" Creates the matrices from the map informations and invokes findandmap method.
	
	"""
	global countSLAMFusion
	name = 'SLAMFusion'
	ErrorsTuple=(1.0,2.0,3.0)
	"""
	a_or = matrixmaker(datagmapping)
	b_or = matrixmaker(datakarto)
	c_or = matrixmaker(datartab)
	"""
	a_or =np.load('./lastframes/No_Obs_SP/AtrioT1/gmapping/4/gmapping4MATRIZ.bin').astype(np.float32)
	b_or =np.load('./lastframes/No_Obs_SP/AtrioT1/karto/64/karto64MATRIZ.bin').astype(np.float32)
	c_or =np.load('./lastframes/No_Obs_SP/AtrioT1/rtabmap/147/rtabmap147MATRIZ.bin').astype(np.float32)
	
	#Calcular os erros considerando a rotação dos mapas.
	erroNobs =[]


	if not os.path.exists('./minusone'):
		os.makedirs('./minusone')
        

	a_or=removeminusone(a_or)
	b_or=removeminusone(b_or)
	c_or=removeminusone(c_or)
    
	shapearray=[]

	shapearray.append(a_or.shape)
	shapearray.append(b_or.shape)
	shapearray.append(c_or.shape)
	newsize = findmaxcoord(shapearray)
		
	a_or=forceToSize(a_or,newsize)
	b_or=forceToSize(b_or,newsize)
	c_or=forceToSize(c_or,newsize)
	
	GmapErro,GmapMap  =calcErrosMK2(a_or,a_or,erroOnly=False)
	
	mkimg(GmapMap,'./gmap.tiff')

	print ("b_or")
	KartoErro,KartoMap =calcErrosMK2(a_or,b_or,erroOnly=False)
	mkimg(KartoMap,'./karto.tiff')

	print ("c_or")
	RTABErro,RTABMap  =calcErrosMK2(a_or,c_or,erroOnly=False)
	mkimg(RTABMap,'./rtab.tiff')
	


	"""Achar o mapa final a partir da média ponderada"""
	print ("findandmapP")
	final = newmapPond(GmapMap,KartoMap,RTABMap,ErrorsTuple)
	finalNorm = newmap(GmapMap,KartoMap,RTABMap)
	mkimg(final,'./finalmap.tiff')
	mkimg(finalNorm,'./finalmapVot.tiff')


def processCycle(array,mapA,mapB,rot,samesize=0):
	""" Auxiliar method for running the matchfinding method.
	
	:param array: Usualy provided by arrayHalfMaker, containing the decimation values.
	:type array: Integer array.
	:param mapA: Contains the map values, from the bigger map, usualy Gmapping or RTABMap.
	:type mapA: Float Matrice.
	:param mapB: Contains the map values, from the smaller map, usualy Karto.
	:type mapB: Float Matrice.
	:returns: Tuple - (Error value,Start coordinate X,Start coordinate Y,Rotation)
	
	"""
	soma=[]
	varx = 0
	vary = 0
	trigger = 0
	prev = 0
	print(samesize,array,'yay')
	if(samesize==1):
		prev =1
		trigger = 1
	for i in array:
		soma=process(varx,vary,i,prev,trigger,mapA,mapB,soma,rot) #START gmap-karto
		varx=soma[0][1]
		vary=soma[0][2]
		trigger = 1
		prev = i
	soma.sort()	
	return soma
	
def topresults(array,name,value=1):
	""" Prints N results from the provided array and the total length.
	
	:param array: Contains the values to be printed.
	:type array: Tuple - (Error value,Start coordinate X,Start coordinate Y,Rotation).
	:param name: Name of the array.
	:type name: String.
	:param value: Value of N values to print.
	:type value: Integer, default:5.
	
	"""
	if(len(array)<value):
		value=len(array)
	for i in range(0,value):
		print(i+1,'-',name,array[i])
	#print(len(array),'\n')

def findandmapV2(a_or,b_or,c_or,debug=False): 
	""" Wrapper method that calls the various methods to create the merged map.
	
	:param array: Usualy provided by arrayHalfMaker, containing the decimation values.
	:type array: Integer array.
	:param a_or: Contains the map values, from one of the bigger maps, usualy Gmapping.
	:type a_or: Float matrix.
	:param b_or: Contains the map values, from one of the bigger maps, usualy RTABMap.
	:type b_or: Float matrix.
	:param c_or: Contains the map values, from the smaller map, usualy Karto.
	:type c_or: Float matrix.
	:param debug: Contains the map values, from the smaller map, usualy Karto.
	:type debug: boolean, default:False.
	:returns: Float matrix containing the final map after all calculations
	
	"""
	#Verifica se as variaveis não estão vazias.
	if (a_or.size and b_or.size and c_or.size):
		
		#crio duas listas vazias.
		somaAB=[]
		somaCB=[]
		
		#Procurar as coordenadas onde os mapas ficam melhor alinhados com translação.
		print ( a_or.shape,c_or.shape,b_or.shape)
		somaAB=processCycle([60,30,15,5,1],a_or,b_or)
		somaCB=processCycle([5,2],c_or,b_or) #START rtab-karto
		
		#Cria se os crops das imagens, baseado nos resultados obtidos com as chamadas as funções process. Com o tamanho do mapa mais pequeno,
		#neste caso o mapa do Karto.
		newmapGM = np.array(a_or[somaAB[0][1]:b_or.shape[0]+somaAB[0][1],somaAB[0][2]:b_or.shape[1]+somaAB[0][2]]).astype(np.float32)
		newmapRT = np.array(c_or[somaCB[0][1]:b_or.shape[0]+somaCB[0][1],somaCB[0][2]:b_or.shape[1]+somaCB[0][2]]).astype(np.float32)
		newmapKT = b_or
		
		rowsA,colsA = newmapGM.shape
		rowsB,colsB = newmapRT.shape
		rowsC,colsC = newmapKT.shape

		#alinhamento por rotação, entre -ROTMAX graus ate ROTMAX graus. (ROTMAX = 5)
		#guarda resultado no vector correspondente.
		#Caso ja estejam alinhados, ao fazer rotação para 0, devolve matriz crop original
		for i in xrange(-ROTMAX,ROTMAX,1):
			A = cv2.getRotationMatrix2D((colsA/2,rowsA/2),i,1)
			C = cv2.getRotationMatrix2D((colsC/2,rowsC/2),i,1)
			rotA = cv2.warpAffine(newmapGM,A,(colsA,rowsA))
			rotC = cv2.warpAffine(newmapRT,C,(colsC,rowsC))
			
			somaAB=process(0,0,1,1,1,rotA,b_or,somaAB,i)
			somaCB=process(0,0,1,1,1,rotC,b_or,somaCB,i)
		

		
		#aplicar parametros da rotação que deu menor erro.
		A = cv2.getRotationMatrix2D((colsA/2,rowsA/2),somaAB[0][3],1)
		C = cv2.getRotationMatrix2D((colsC/2,rowsC/2),somaCB[0][3],1)
		newmapGMROT = np.array(cv2.warpAffine(newmapGM,A,(colsA,rowsA))).astype(np.float32)
		newmapRTROT = np.array(cv2.warpAffine(newmapRT,C,(colsC,rowsC))).astype(np.float32)		

		#Chamar função para fazer votação dos 3 mapas obtidos.
		final = newmap(newmapGMROT,newmapRTROT,newmapKT)
		#mkimg(final,'finalmap.tiff')
		if(debug):
			np.savetxt('./'+name+'/'+str(countSLAMFusion)+'/'+'final'+str(countSLAMFusion)+'.txt',final,fmt='%.0f')
			mkimg(final,'./'+name+'/'+str(countSLAMFusion)+'/'+'finalmap'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapGM,'./'+name+'/'+str(countSLAMFusion)+'/'+'gmcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapRT,'./'+name+'/'+str(countSLAMFusion)+'/'+'rtcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapKT,'./'+name+'/'+str(countSLAMFusion)+'/'+'ktcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapGMROT,'./'+name+'/'+str(countSLAMFusion)+'/'+'gmapROT'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapRTROT,'./'+name+'/'+str(countSLAMFusion)+'/'+'rtabROT'+str(countSLAMFusion)+'.tiff')
		countSLAMFusion +=1
		return final
		
def findandmapP(a_or,b_or,c_or,errors,debug=True): 
	stepvar = 1
	samesize = 1
	rtabmap =  calcErros(c_or,b_or,stepvar,rotstep=1,samesize=samesize)
	gmapping = calcErros(a_or,b_or,stepvar,rotstep=1,samesize=samesize)

	
	gmappingcrop = np.array(a_or[gmapping[1]:b_or.shape[0]+gmapping[1],gmapping[2]:b_or.shape[1]+gmapping[2]]).astype(np.float32)
	rtabmapcrop = np.array(c_or[rtabmap[1]:b_or.shape[0]+rtabmap[1],rtabmap[2]:b_or.shape[1]+rtabmap[2]]).astype(np.float32)
	
	mkimg(gmappingcrop   ,'./gmapcrop.tiff')
	mkimg(rtabmapcrop   ,'./rtabcrop.tiff')	
	
	rowsA,colsA = b_or.shape
	A = cv2.getRotationMatrix2D((colsA/2,rowsA/2),gmapping[3],1)
	C = cv2.getRotationMatrix2D((colsA/2,rowsA/2),rtabmap[3],1)
	newmapGMROT = np.array(cv2.warpAffine(gmappingcrop,A,(colsA,rowsA),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS,borderValue=(-1, -1, -1, -1))).astype(np.float32)
	newmapRTROT = np.array(cv2.warpAffine(rtabmapcrop,C,(colsA,rowsA),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS,borderValue=(-1, -1, -1, -1))).astype(np.float32)	
	#rotA = cv2.warpAffine(a_or,A,(colsA,rowsA),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS,borderValue=(-1, -1, -1, -1))	
	#rotA = cv2.warpAffine(a_or,A,(colsA,rowsA),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS,borderValue=(-1, -1, -1, -1))	
	mkimg(newmapGMROT   ,'./gmap_crop_rot.tiff')
	mkimg(newmapRTROT   ,'./rtab_crop_rot.tiff')	
	final = newmapPond(newmapGMROT,newmapRTROT,b_or,(errors[0],errors[1],errors[2]))
	
	return final
	
	
def findandmapPV1(a_or,b_or,c_or,errors,debug=True): 
	""" Wrapper method that calls the various methods to create the merged map.
	
	:param array: Usualy provided by arrayHalfMaker, containing the decimation values.
	:type array: Integer array.
	:param a_or: Contains the map values, from one of the bigger maps, usualy Gmapping.
	:type a_or: Float matrix.
	:param b_or: Contains the map values, from one of the bigger maps, usualy RTABMap.
	:type b_or: Float matrix.
	:param c_or: Contains the map values, from the smaller map, usualy Karto.
	:type c_or: Float matrix.
	:param debug: Contains the map values, from the smaller map, usualy Karto.
	:type debug: boolean, default:False.
	:returns: Float matrix containing the final map after all calculations
	
	"""
	#Verifica se as variaveis não estão vazias.
	if (a_or.size and b_or.size and c_or.size):
		
		#crio duas listas vazias.
		somaAB=[]
		somaCB=[]
		
		#Procurar as coordenadas onde os mapas ficam melhor alinhados com translação.
		print ( a_or.shape,c_or.shape,b_or.shape,"findandmapP")
		somaAB=processCycle(arrayHalfMaker(100),a_or,b_or)
		somaCB=processCycle(arrayHalfMaker(20),c_or,b_or) #START rtab-karto
		
		#Cria se os crops das imagens, baseado nos resultados obtidos com as chamadas as funções process. Com o tamanho do mapa mais pequeno,
		#neste caso o mapa do Karto.
		newmapGM = np.array(a_or[somaAB[0][1]:b_or.shape[0]+somaAB[0][1],somaAB[0][2]:b_or.shape[1]+somaAB[0][2]]).astype(np.float32)
		newmapRT = np.array(c_or[somaCB[0][1]:b_or.shape[0]+somaCB[0][1],somaCB[0][2]:b_or.shape[1]+somaCB[0][2]]).astype(np.float32)
		newmapKT = b_or
		
		rowsA,colsA = newmapGM.shape
		rowsB,colsB = newmapRT.shape
		rowsC,colsC = newmapKT.shape

		#alinhamento por rotação, entre -ROTMAX graus ate ROTMAX graus. (ROTMAX = 5)
		#guarda resultado no vector correspondente.
		#Caso ja estejam alinhados, ao fazer rotação para 0, devolve matriz crop original
		for i in xrange(-ROTMAX,ROTMAX,1):
			A = cv2.getRotationMatrix2D((colsA/2,rowsA/2),i,1)
			C = cv2.getRotationMatrix2D((colsC/2,rowsC/2),i,1)
			rotA = cv2.warpAffine(newmapGM,A,(colsA,rowsA))
			rotC = cv2.warpAffine(newmapRT,C,(colsC,rowsC))
			
			somaAB=process(0,0,1,1,1,rotA,b_or,somaAB,i)
			somaCB=process(0,0,1,1,1,rotC,b_or,somaCB,i)
		

		
		#aplicar parametros da rotação que deu menor erro.
		A = cv2.getRotationMatrix2D((colsA/2,rowsA/2),somaAB[0][3],1)
		C = cv2.getRotationMatrix2D((colsC/2,rowsC/2),somaCB[0][3],1)
		newmapGMROT = np.array(cv2.warpAffine(newmapGM,A,(colsA,rowsA))).astype(np.float32)
		newmapRTROT = np.array(cv2.warpAffine(newmapRT,C,(colsC,rowsC))).astype(np.float32)		

		#Chamar função para fazer média ponderada dos 3 mapas obtidos.
		final = newmapPond(newmapGMROT,newmapRTROT,newmapKT,(errors[0],errors[1],errors[2]))
		#mkimg(final,'finalmap.tiff')
		countSLAMFusion = 1
		if(debug):
			np.savetxt('./2/'+'final'+str(countSLAMFusion)+'.txt',final,fmt='%.0f')
			mkimg(final,'./2/'+'finalmap'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapGM,'./2/'+'gmcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapRT,'./2/'+'rtcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapKT,'./2/'+'ktcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapGMROT,'./2/'+'gmapROT'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapRTROT,'./2/'+'rtabROT'+str(countSLAMFusion)+'.tiff')
		#countSLAMFusion +=1
		return final

def calcErrosMK1(a_or,b_or,erroOnly=False): 
	""" Wrapper method that calls the various methods to create the merged map.
	
	:param array: Usualy provided by arrayHalfMaker, containing the decimation values.
	:type array: Integer array.
	:param a_or: Contains the map values, from one of the bigger maps, usualy Gmapping.
	:type a_or: Float matrix.
	:param b_or: Contains the map values, from one of the bigger maps, usualy RTABMap.
	:type b_or: Float matrix.
	:param c_or: Contains the map values, from the smaller map, usualy Karto.
	:type c_or: Float matrix.
	:param debug: Contains the map values, from the smaller map, usualy Karto.
	:type debug: boolean, default:False.
	:returns: Float matrix containing the final map after all calculations
	
	"""
	#Verifica se as variaveis não estão vazias.
	if (a_or.size and b_or.size):
		if(not erroOnly):
				
			#im1 =  cv2.imread(mkimg(a_or,"./tmp/a_or.tiff"));
			#im2 =  cv2.imread(mkimg(b_or,"./tmp/b_or.tiff"));
			
			# Convert images to grayscale
			#im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
			#im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
			# Find size of image1
			sz = a_or.shape
		 
			# Define the motion model
			warp_mode = cv2.MOTION_EUCLIDEAN
			#warp_mode = cv2.MOTION_AFFINE
			
			warp_matrix = np.eye(2, 3, dtype=np.float32)    
			 
			# Specify the number of iterations.
			number_of_iterations = 5000; #5000
			 
			# Specify the threshold of the increment
			# in the correlation coefficient between two iterations
			termination_eps = 1e-10; #1e-10
			
			# Define termination criteria
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
			 
			# Run the ECC algorithm. The results are stored in warp_matrix.
			(cc, warp_matrix) = cv2.findTransformECC (a_or,b_or,warp_matrix, warp_mode, criteria)
			
			im2_aligned = cv2.warpAffine(b_or, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS,borderValue=(-1,-1,-1,-1));
		
		finsoma = []
		#finsoma=process(0,0,1,1,1,a_or,im2_aligned,finsoma,'Unk')
		print('Process!')
		""""
		if(not erroOnly):
			finsoma=process(a_or,im2_aligned)
		else:
			finsoma=process(a_or,b_or)
		"""
		if(not erroOnly):
			finsoma=processMK1(a_or,im2_aligned)
		else:              
			finsoma=processMK1(a_or,b_or)
		#topresults(soma,'var',value=5)
		#os.system("rm ./out_registration.txt ./c++/temp*.tiff")
		"""
		mkimg(a_or,"./tmp/a_or.tiff")
		mkimg(b_or,"./tmp/b_or.tiff")
		mkimg(im2_aligned,"./tmp/b_or_aligned.tiff")
		print('wait here')
		raw_input()
		"""
		if(not erroOnly):
			print("yay")
			return (finsoma,im2_aligned)
		return finsoma

def calcErrosMK2(a_or,b_or,erroOnly=False):
	# Read the images to be aligned
	mkimg(a_or,"./tmp/a_or.tiff")
	mkimg(b_or,"./tmp/b_or.tiff")
	im1 =  cv2.imread("./tmp/a_or.tiff");
	im2 =  cv2.imread("./tmp/b_or.tiff");
	 
	# Convert images to grayscale
	im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
	 
	# Find size of image1
	sz = im1.shape
	 
	# Define the motion model
	warp_mode = cv2.MOTION_TRANSLATION
	 
	# Define 2x3 or 3x3 matrices and initialize the matrix to identity
	if warp_mode == cv2.MOTION_HOMOGRAPHY :
	    warp_matrix = np.eye(3, 3, dtype=np.float32)
	else :
	    warp_matrix = np.eye(2, 3, dtype=np.float32)
	 
	# Specify the number of iterations.
	number_of_iterations = 5000;
	 
	# Specify the threshold of the increment
	# in the correlation coefficient between two iterations
	termination_eps = 1e-10;
	 
	# Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
	 
	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
	 
	    # Use warpAffine for Translation, Euclidean and Affine
	#im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
	im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS,borderValue=(240,240,240,240));
	    
	finsoma = []
	#finsoma=process(0,0,1,1,1,a_or,im2_aligned,finsoma,'Unk')
	print('Process!')
        im2_aligned=treatimg(im2_aligned)
	if(not erroOnly):
		print(a_or.shape,im2_aligned.shape)
		#finsoma=processMK1(a_or,im2_aligned)
		finsoma=erro(a_or,im2_aligned)
	else:              
		#finsoma=processMK1(a_or,b_or)
		finsoma=erro(a_or,b_or)
	#topresults(soma,'var',value=5)
	#os.system("rm ./out_registration.txt ./c++/temp*.tiff")
	"""
	for i in range(0,600):
		for j in range(0,600):
			if(finsoma[1][i][j]!=finsoma2[1][i][j]):
				print((i,j),(finsoma[1][i][j],finsoma2[1][i][j]),(a_or[i][j],im2_aligned[i][j]))
				time.sleep(5)
	"""
	if(not erroOnly):
            return (finsoma,im2_aligned)
	return finsoma
    
    
def calcErrosV2(a_or,b_or,halfmaker=20,samesize=0,debug=False,rotstep=1): 
	""" Wrapper method that calls the various methods to create the merged map.
	
	:param array: Usualy provided by arrayHalfMaker, containing the decimation values.
	:type array: Integer array.
	:param a_or: Contains the map values, from one of the bigger maps, usualy Gmapping.
	:type a_or: Float matrix.
	:param b_or: Contains the map values, from one of the bigger maps, usualy RTABMap.
	:type b_or: Float matrix.
	:param c_or: Contains the map values, from the smaller map, usualy Karto.
	:type c_or: Float matrix.
	:param debug: Contains the map values, from the smaller map, usualy Karto.
	:type debug: boolean, default:False.
	:returns: Float matrix containing the final map after all calculations
	
	"""
	#Verifica se as variaveis não estão vazias.
	if (a_or.size and b_or.size):
		
		#crio duas listas vazias.
		soma=[]
		somaaux=[]
		#Procurar as coordenadas onde os mapas ficam melhor alinhados com translação.
		print(arrayHalfMaker(halfmaker),samesize)
		soma=processCycle(arrayHalfMaker(halfmaker),a_or,b_or,0,samesize=samesize)
		soma.sort()
		a_crop = np.array(a_or[soma[0][1]:b_or.shape[0]+soma[0][1],soma[0][2]:b_or.shape[1]+soma[0][2]]).astype(np.float32)
		rowsA,colsA = a_crop.shape
		bestposX =soma[0][1]
		bestposY =soma[0][2]
		print(bestposX,bestposY)
		#mkimg(a_crop,'./fin/a_crop_pre-rots.tiff')
		mkimg(b_or,'./fin/b_or_pre-rots.tiff')
		#for i in xrange(-ROTMAX,ROTMAX,rotstep):
		topresults(soma,'Pre',value=1)
		
		# Find size of image1
		sz = a_crop.shape
	 
		# Define the motion model
		warp_mode = cv2.MOTION_EUCLIDEAN
		
		warp_matrix = np.eye(2, 3, dtype=np.float32)    
		 
		# Specify the number of iterations.
		number_of_iterations = 5000;
		 
		# Specify the threshold of the increment
		# in the correlation coefficient between two iterations
		termination_eps = 1e-10;
		
		# Define termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
		 
		# Run the ECC algorithm. The results are stored in warp_matrix.
		(cc, warp_matrix) = cv2.findTransformECC (a_crop,b_or,warp_matrix, warp_mode, criteria)
		
		im2_aligned = cv2.warpAffine(b_or, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS,borderValue=(-1,-1,-1,-1));
		print (im2_aligned.shape)
		finsoma = []
		finsoma=process(0,0,1,1,1,a_crop,im2_aligned,finsoma,404,bestposX,bestposY,1)
		mkimg(im2_aligned,'./fin/im2_align.tiff')
		mkimg(a_crop,'./fin/acrop.tiff')
		finsoma.sort()
		topresults(finsoma,'Pos',value=len(finsoma))
		""""
		for i in xrange(-ROTMAX,ROTMAX,rotstep):
			print(i)
			sys.stdout.flush()
			rowsB,colsB = b_or.shape
			#B = cv2.getRotationMatrix2D((colsB/2,rowsB/2),i,1)
			print(B)
			rotB = cv2.warpAffine(b_or,B,(colsB,rowsB),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS,borderValue=(-1, -1, -1, -1))	
			#somaaux=processCycle(arrayHalfMaker(1),a_crop,rotB,i,1)
			mkimg(rotB,'./rot/'+str(i)+'B.tiff')
			mkimg(a_crop,'./rot/'+str(i)+'A.tiff')
			#soma=soma+somaaux
			soma=process(0,0,1,1,1,a_crop,rotB,soma,i,bestposX,bestposY,1)
		"""""
		

		

		#Cria se os crops das imagens, baseado nos resultados obtidos com as chamadas as funções process. Com o tamanho do mapa mais pequeno,
		#neste caso o mapa do Karto.

		#alinhamento por rotação, entre -ROTMAX graus ate ROTMAX graus. (ROTMAX = 5)
		#guarda resultado no vector correspondente.
		#Caso ja estejam alinhados, ao fazer rotação para 0, devolve matriz crop original
			
		
		if(debug):
			countSLAMFusion = 1
			np.savetxt('./2/'+'final'+str(countSLAMFusion)+'.txt',final,fmt='%.0f')
			mkimg(final,'./2/'+'finalmap'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapGM,'./2/'+'gmcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapRT,'./2/'+'rtcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapKT,'./2/'+'ktcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapGMROT,'./2/'+'gmapROT'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapRTROT,'./2/'+'rtabROT'+str(countSLAMFusion)+'.tiff')
		#countSLAMFusion +=1
		
		soma.sort()
		#topresults(soma,'var',value=5)
		#os.system("rm ./out_registration.txt ./c++/temp*.tiff")
		return (finsoma[0],im2_aligned)
					
def calcErros(a_or,b_or,halfmaker=20,debug=False): 
	""" Wrapper method that calls the various methods to create the merged map.
	
	:param array: Usualy provided by arrayHalfMaker, containing the decimation values.
	:type array: Integer array.
	:param a_or: Contains the map values, from one of the bigger maps, usualy Gmapping.
	:type a_or: Float matrix.
	:param b_or: Contains the map values, from one of the bigger maps, usualy RTABMap.
	:type b_or: Float matrix.
	:param c_or: Contains the map values, from the smaller map, usualy Karto.
	:type c_or: Float matrix.
	:param debug: Contains the map values, from the smaller map, usualy Karto.
	:type debug: boolean, default:False.
	:returns: Float matrix containing the final map after all calculations
	
	"""
	#Verifica se as variaveis não estão vazias.
	if (a_or.size and b_or.size):
		
		#crio duas listas vazias.
		somaAB=[]
		
		#Procurar as coordenadas onde os mapas ficam melhor alinhados com translação.
		var = 0
		if(halfmaker==1):
			var=1

		somaAB=processCycle(arrayHalfMaker(halfmaker),a_or,b_or,0,samesize=var)

		#Cria se os crops das imagens, baseado nos resultados obtidos com as chamadas as funções process. Com o tamanho do mapa mais pequeno,
		#neste caso o mapa do Karto.
		a_crop = np.array(a_or[somaAB[0][1]:b_or.shape[0]+somaAB[0][1],somaAB[0][2]:b_or.shape[1]+somaAB[0][2]]).astype(np.float32)
		rowsA,colsA = a_crop.shape
		bestposX =somaAB[0][1]
		bestposY =somaAB[0][2]
		print(bestposX,bestposY)
		#alinhamento por rotação, entre -ROTMAX graus ate ROTMAX graus. (ROTMAX = 5)
		#guarda resultado no vector correspondente.
		#Caso ja estejam alinhados, ao fazer rotação para 0, devolve matriz crop original

		for i in xrange(-ROTMAX,ROTMAX,1):
			A = cv2.getRotationMatrix2D((colsA/2,rowsA/2),i,1)
			rotA = cv2.warpAffine(a_crop,A,(colsA,rowsA),flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS,borderValue=(-1, -1, -1, -1))	
			#somaAB=processCycle(arrayHalfMaker(1),rotA,b_or,1)
			somaAB=process(0,0,1,1,1,rotA,b_or,somaAB,i,bestposX,bestposY,1)
			#mkimg(rotA,'./rot/'+str(i)+'.tiff')
			
		
		if(debug):
			countSLAMFusion = 1
			np.savetxt('./2/'+'final'+str(countSLAMFusion)+'.txt',final,fmt='%.0f')
			mkimg(final,'./2/'+'finalmap'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapGM,'./2/'+'gmcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapRT,'./2/'+'rtcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapKT,'./2/'+'ktcrop'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapGMROT,'./2/'+'gmapROT'+str(countSLAMFusion)+'.tiff')
			mkimg(newmapRTROT,'./2/'+'rtabROT'+str(countSLAMFusion)+'.tiff')
		#countSLAMFusion +=1
		somaAB.sort()
		return somaAB[0]
			
def erro(a,b):
        nc, nl = a.shape
        errmat = np.zeros_like(a)
        e=0.0
        for i in range(nc):
                for j in range(nl):
                        if(a[i][j] != b[i][j]):
                                e=e+1
        #print("testes!")
        #print (e)
        #print(float("{0:.2f}".format(e)))
        #print (nc,nl,nc*nl)
        #print(float("{0:.4f}".format(e/(nc*nl))))
        #print("end testes!")
        r = float("{0:.5f}".format(e/(nc*nl)))
        #print(r)
        return r
		
		
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
mf = cl.mem_flags
prg = cl.Program(ctx, """
  /*int row = get_global_id(0);
  int col = get_global_id(1);
  int address = ((row)*(numcola[0])) + (col);*/
	//j =j+(a[count]-b[count])*(a[count]-b[count]);
	//j =j+fabs(a[count]-b[count]);
	
__kernel void quadraticdif(__global const int *a, __global const int *b,__global int *resfloat,__global int * numcola) {
  int locid= get_local_id(0);
  if(locid==0)
  {
  int totala = numcola[0]*numcola[1];
  int count;
  float j=0;
  for(count=0;count<totala;count++){

	if(abs(a[count]-b[count]) != 0)
	{
		resfloat[count]=1;
		
	}
	else
		resfloat[count]=0;
  }
  }
}
__kernel void clr(__global int *res,__global int * numcola) {
  int totala = numcola[0]*numcola[1];
  int count;
  float j=0;
  for(count=0;count<totala;count++){
	res[count] = NULL;
	}
}

__kernel void average(__global const float *a, __global const float *b,__global const float *c, __global float *res,__global int * numcola) {
  int totala = numcola[0]*numcola[1];
  int count;
  float j=0;
  for(count=0;count<totala;count++){
	
	if (a[count] == b[count] || a[count]==c[count])
        res[count]=a[count];
	else if (b[count] == c[count])
        res[count]=b[count];
    else res[count]=-1;
    //else res[count]=round((a[count]+b[count]+c[count])/3.0);

  }
}
  
__kernel void ponderedaverageV1(__global const float *a, __global const float *b,__global const float *c, __global float *res,__global int * numcola,__global float *errors) {
  int totala = numcola[0]*numcola[1];
  float e1 = errors[0],e2 = errors[1],e3 = errors[2];
  float s = e1+e2+e3;
  float s1 = s-e1,  s2 = s-e2, s3 = s-e3;
  s *= 2;
  int count;
  float j=0;
  for(count=0;count<totala;count++){
	res[count]=round( ( (s1*a[count]) + (s2*b[count]) + (s3*c[count])) / s );
  }

}

__kernel void ponderedaverage(__global const float *a, __global const float *b,__global const float *c, __global float *res,__global int * numcola,__global float *errors) {
  int totala = numcola[0]*numcola[1];
  float e1 = errors[0],e2 = errors[1],e3 = errors[2];
  float s = e1+e2+e3;
  float s1 = s-e1,  s2 = s-e2, s3 = s-e3;
  float w1,w2,w3;
  //float p1=7 , p2=2 ,p3=1; old
  float p1=7 , p2=2 ,p3=1;
  //float p1=errors[0], p2=errors[1] ,p3=errors[2];
  
  float div = p1+p2+p3;
  
  if(s1 > s2 && s1>s3 && s2>s3) // 123  (antes >)
  {
	w1=p1;
	w2=p2;
	w3=p3;
  }
  else if(s1 > s2 && s1>s3 && s3>s2) // 132
  {
	w1=p1;
	w2=p3;
	w3=p2;
  }
  else if(s2 > s1 && s2>s3 && s1>s3) // 213
  {
	w1=p2;
	w2=p1;
	w3=p3;
  }
   else if(s2 > s1 && s2>s3 && s3>s1) // 231
  {
	//w1=p3;
	//w2=p1;
	//w3=p2;
        w1=p2;
        w2=p3;
        w3=p1;
  }
 
   else if(s3> s1 && s3>s2 && s1>s2) // 312
  {
	//w1=p2;
	//w2=p3;
	//w3=p1;
        w1=p3;
        w2=p1;
        w3=p2;
  }
   else if(s3 > s1 && s3>s2 && s2>s1) // 321
  {
	w1=p3;
	w2=p2;
	w3=p1;
  }
  
  
  int count;
  float j=0;
  for(count=0;count<totala;count++){
	res[count]=round( ( (w1*a[count]) + (w2*b[count]) + (w3*c[count])) / div );
  }

}

""").build()

def processMK1(a_np,b_np):
	""" Wrapper method that calls the various methods to create the merged map.
	
	:param a_np: Contains the map values, from one of the bigger maps, usualy Gmapping.
	:type a_np: Float matrix.
	:param b_np: Contains the map values, from the smaller map, usualy Karto.
	:type b_np: Float matrix.
	:returns: The original Tuple array with the new calculations.
	
	"""

	#print (valX, 'in', rang1-1, 'and', valY,'in',rang2-1,'                      ', end='\r')	
	### INPUT VARS ###)
	a_np = np.array(a_np).astype(np.int32)
	b_np = np.array(b_np).astype(np.int32)
	
	numcola = np.array(a_np.shape).astype(np.int32) # create array that is the tuple a_np-size (X,Y)
	### INPUT VARS ###
	
	### TEST PRINTS ###
	### TEST PRINTS ###
	
	### INPUT BUFFERS ###
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
	b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
	numcola_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numcola)
	### INPUT BUFFERS ###
	
	
	### OUTPUT VARS ###
	resint_np = np.zeros_like(a_np).astype(np.int32)
	### OUTPUT VARS ###
	
	### OUTPUT BUFFERS ###
	resint_buf = cl.Buffer(ctx, mf.WRITE_ONLY, resint_np.nbytes)
	### OUTPUT BUFFERS ###
	
	###EXECUTE###

	exe_bla = prg.quadraticdif(queue, a_np.shape, None, a_buf, b_buf,resint_buf,numcola_buf)
	cl.enqueue_copy(queue, resint_np, resint_buf)
	
	###EXECUTE###
        print("process",sum(sum(resint_np)))
	return (sum(sum(resint_np)),resint_np)

def process(varx,vary,step,prestep,trigger,firstmap,secmap,soma,rot,rotx=0,roty=0,rotTri=0):
	""" Wrapper method that calls the various methods to create the merged map.
	
	:param varx: Start value for coordinate X.
	:type varx: Integer.
	:param vary: Start value for coordinate Y.
	:type vary: Integer.
	:param step: Step value for the the cycles.
	:type step: Integer.
	:param prestep: Step value for the previous call of the method, At start is 0.
	:type prestep: Integer.
	:param trigger: A trigger value, at first run it is 0, following runs it is 1.
	:type trigger: Integer.
	:param firstmap: Contains the map values, from one of the bigger maps, usualy Gmapping.
	:type firstmap: Float matrix.
	:param secmap: Contains the map values, from the smaller map, usualy Karto.
	:type secmap: Float matrix.
	:param soma: The array where the calculations are saved.
	:type soma: Tuple array with values (Error value,Start coordinate X,Start coordinate Y,Rotation).
	:param rot: Rotation value.
	:type rot: Integer.
	:returns: The original Tuple array with the new calculations.
	
	"""
	varx=varx-prestep
	if(varx<0):
		varx=0
	vary=vary-prestep
	if(vary<0):
		vary=0
	if(trigger==0):
		rang1=firstmap.shape[0]-secmap.shape[0]
		rang2=firstmap.shape[1]-secmap.shape[1]

	else:
		rang1=varx+prestep*2
		rang2=vary+prestep*2
		
	if(rang1<0):
		rang1=1
	if(rang2<0):
		rang2=1
		
	for valX in xrange(varx,rang1,step):
		for valY in xrange(vary,rang2,step):
			print (valX, 'in', rang1-1, 'and', valY,'in',rang2-1,'                      ', end='\r')	
			### INPUT VARS ###)
			b_np = np.array(secmap).astype(np.float32)
			a_np = np.array(firstmap[valX:secmap.shape[0]+valX,valY:secmap.shape[1]+valY]).astype(np.float32)
			
			numcola = np.array(a_np.shape).astype(np.int32) # create array that is the tuple a_np-size (X,Y)
			### INPUT VARS ###
			
			### TEST PRINTS ###
			### TEST PRINTS ###
			
			### INPUT BUFFERS ###
			a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
			b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
			numcola_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numcola)
			### INPUT BUFFERS ###
			
			
			### OUTPUT VARS ###
			resint_np = np.array([0]).astype(np.float32)
			### OUTPUT VARS ###
			
			### OUTPUT BUFFERS ###
			resint_buf = cl.Buffer(ctx, mf.WRITE_ONLY, resint_np.nbytes)
			### OUTPUT BUFFERS ###
			
			###EXECUTE###

			exe_bla = prg.quadraticdif(queue, a_np.shape, None, a_buf, b_buf,resint_buf,numcola_buf)
			cl.enqueue_copy(queue, resint_np, resint_buf)
			#print(a_np.size)
			#print(a_np.shape)
			if(rotTri==1):
				soma.append((resint_np[0],rotx,roty,rot))
			else:            
				soma.append((resint_np[0],valX,valY,rot))
			#print(1e-9*(exe_bla.profile.end-exe_bla.profile.start))
			###EXECUTE
	soma.sort()

	return soma

def newmap(Fmap,Smap,Tmap):	
	""" Method that creates the final merged map.
	
	:param trigger: A trigger value, at first run it is 0, following runs it is 1.
	:type trigger: Integer.
	:param Fmap: Contains the map values, from one of the bigger maps, usualy Gmapping.
	:type Fmap: Float matrix.
	:param Smap: Contains the map values, from the smaller map, usualy Karto.
	:type Smap: Float matrix.
	:param Tmap: Contains the map values, from the smaller map, usualy Karto.
	:type Tmap: Float matrix.
	:returns: The final map-merged matrix.
	
	"""
	a_np = np.array(Fmap).astype(np.float32)
	b_np = np.array(Smap).astype(np.float32)
	c_np = np.array(Tmap).astype(np.float32)
	numcola = np.array(a_np.shape).astype(np.int32)
	### INPUT VARS ###
	
	### TEST PRINTS ###
	### TEST PRINTS ###
	
	### INPUT BUFFERS ###
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
	b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
	c_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c_np)
	numcola_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numcola)
	### INPUT BUFFERS ###
	
	
	### OUTPUT VARS ###
	resint_np = np.array(a_np).astype(np.float32)
	### OUTPUT VARS ###
	
	### OUTPUT BUFFERS ###
	resint_buf = cl.Buffer(ctx, mf.WRITE_ONLY, resint_np.nbytes)
	### OUTPUT BUFFERS ###
	
	###EXECUTE###
	prg.clr(queue, a_np.shape, None,resint_buf,numcola_buf)
	cl.enqueue_copy(queue, resint_np, resint_buf)
	
	exe_bla =prg.average(queue, a_np.shape, None, a_buf, b_buf,c_buf,resint_buf,numcola_buf)
	cl.enqueue_copy(queue, resint_np, resint_buf)
	return resint_np
	#print(1e-9*(exe_bla.profile.end-exe_bla.profile.start))
	###EXECUTE

def newmapPond(Fmap,Smap,Tmap,erros):	
	""" Method that creates the final merged map.
	
	:param trigger: A trigger value, at first run it is 0, following runs it is 1.
	:type trigger: Integer.
	:param Fmap: Contains the map values, from one of the bigger maps, usualy Gmapping.
	:type Fmap: Float matrix.
	:param Smap: Contains the map values, from the smaller map, usualy Karto.
	:type Smap: Float matrix.
	:param Tmap: Contains the map values, from the smaller map, usualy Karto.
	:type Tmap: Float matrix.
	:returns: The final map-merged matrix.
	
	"""
	print(erros)
	s1=erros[0]
	s2=erros[1]
	s3=erros[2]
	p1=1.0
	p2=0.0
	p3=0.0
	if(s1 > s2 and s1>s3 and s2>s3): # 123  (antes >)
		w1=p3
		w2=p2
		w3=p1
		print("123")
	elif(s1 > s2 and s1>s3 and s3>s2): # 132
		w1=p2
		w2=p3
		w3=p1
		print("132")
	elif(s2 > s1 and s2>s3 and s1>s3): # 213
		w1=p3
		w2=p1
		w3=p2
		print("213")
	elif(s2 > s1 and s2>s3 and s3>s1): # 231
	#w1=p3;
	#w2=p1;
	#w3=p2;
		w1=p1
		w2=p3
		w3=p2
		print("231")
	elif(s3> s1 and s3>s2 and s1>s2): # 312
	#w1=p2;
	#w2=p3;
	#w3=p1;
		w1=p2
		w2=p1
		w3=p3
		print("312")
	elif(s3 > s1 and s3>s2 and s2>s1): # 321
		w1=p1
		w2=p2
		w3=p3
		print("321")
	

	
	a_np = np.array(Fmap).astype(np.float32)
	b_np = np.array(Smap).astype(np.float32)
	c_np = np.array(Tmap).astype(np.float32)
	numcola = np.array(a_np.shape).astype(np.int32)
	#errors = np.array([w1,w2,w3]).astype(np.float32)
	errors = np.array([erros[0],erros[1],erros[2]]).astype(np.float32)
	print(errors)
	### INPUT VARS ###
	
	### TEST PRINTS ###
	### TEST PRINTS ###
	
	### INPUT BUFFERS ###
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
	b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
	c_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c_np)
	numcola_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numcola)
	errors_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=errors)
	### INPUT BUFFERS ###
	
	
	### OUTPUT VARS ###
	resint_np = np.array(a_np).astype(np.float32)
	### OUTPUT VARS ###
	
	### OUTPUT BUFFERS ###
	resint_buf = cl.Buffer(ctx, mf.WRITE_ONLY, resint_np.nbytes)
	### OUTPUT BUFFERS ###
	
	###EXECUTE###
	prg.clr(queue, a_np.shape, None,resint_buf,numcola_buf)
	cl.enqueue_copy(queue, resint_np, resint_buf)
	
	exe_bla =prg.ponderedaverage(queue, a_np.shape, None, a_buf, b_buf,c_buf,resint_buf,numcola_buf,errors_buf)
	cl.enqueue_copy(queue, resint_np, resint_buf)
	return resint_np
	#print(1e-9*(exe_bla.profile.end-exe_bla.profile.start))
	###EXECUTE

#listner()
calcmap()
#newmapPond(1,2,3,(0.0588, 0.0566, 0.063))
#print("yay")

