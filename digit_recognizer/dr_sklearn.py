from numpy import *
import csv

def loadTrainData():
	l=[]
	with open('train.csv') as file:
		lines=csv.reader(file)
		for line in lines:
			l.append(line) #42001*785
	l.remove(l[0])
	l=array(l)
	label=l[:,0]
	data=l[:,1:]
	return nomalizing(toInt(data),toInt(label))

def toInt(array):
	array=mat(array)
	m,n=shape(array)
	newArray=zeros((m,n))
	for i in xrange(m):
		for j in xrange(n):
			newArray[i,j]=int(array[i,j])
	return newArray

def nomalizing(array):
	m,n=shape(array)
	for i in xrange(m):
		for j in xrange(n):
			if array[i,j]!=0:
				array[i,j]=1
	return array

def loadTestData():
	l=[]
	with open('test.csv') as file:
		lines=csv.reader(file)
		for line in lines:
			l.append(line) #28001*784
	l.remove(l[0])
	data=array(l)
	return nomalizing(toInt(data))

def loadTestResult():
	l=[]
	with open('knn_benchmark.csv') as file:
		lines=csv.reader(file)
		for line in lines:
			l.append(line) #28001*2
	l.remove(l[0])
	label=array(l[:,1])
	return toInt(label)

def saveResult(result,csvName):
	with open(csvName,'wb') as myfile:
		myWriter=csv.writer(myfile)
		for i in result:
			tmp=[]
			tmp.append(i)
			myWriter.writrow(tmp)

#use scikit-learn package
from sklearn.neighbors import KNeighborsClassifier
def knnClassify(trainData,trainLabel,testData):
	knnClf=KNeighborsClassifier() #default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
	knnClf.fit(trainData,ravel(trainLabel))
	testLabel=knnClf.predict(testData)
	saveResult(testLabel,'sklean_knn_Result.csv')
	return testLabel

from sklearn.neighbors import KNeighborsClassifier  
def knnClassify(trainData,trainLabel,testData): 
	knnClf=KNeighborsClassifier()#default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
	knnClf.fit(trainData,ravel(trainLabel))
	testLabel=knnClf.predict(testData)
	saveResult(testLabel,'sklearn_knn_Result.csv')
	return testLabel

from sklearn.naive_bayes import GaussianNB      #nb for 高斯分布的数据
def GaussianNBClassify(trainData,trainLabel,testData): 
	nbClf=GaussianNB()          
	nbClf.fit(trainData,ravel(trainLabel))
	testLabel=nbClf.predict(testData)
	saveResult(testLabel,'sklearn_GaussianNB_Result.csv')
	return testLabel

from sklearn.naive_bayes import MultinomialNB   #nb for 多项式分布的数据    
def MultinomialNBClassify(trainData,trainLabel,testData): 
	nbClf=MultinomialNB(alpha=0.1)      #default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing.       
	nbClf.fit(trainData,ravel(trainLabel))
	testLabel=nbClf.predict(testData)
	saveResult(testLabel,'sklearn_MultinomialNB_alpha=0.1_Result.csv')
	return testLabel

def printDifferent(ResultName,result,resultGiven):
	m,n=shape(result)
	different=0
	for i in xrange(m):
		if resultGiven[0,i]!=result[i]:
			different+=1
	print ResultName+different

def digitRecognition():
	trainData,trainLabel=loadTrainData()
	testData=loadTestData()
	result1=knnClassify(trainData,trainLabel,testData)
	result2=svcClassify(trainData,trainLabel,testData)
	result3=GaussianNBClassify(trainData,trainLabel,testData)
	result4=MultinomialNBClassify(trainData,trainLabel,testData)
								    
	resultGiven=loadTestResult()
	printDifferent('knn',result1,resultGiven)
	printDifferent('svm',result2,resultGiven)
	printDifferent('nbg',result3,resultGiven)
	printDifferent('nbm',result4,resultGiven)
