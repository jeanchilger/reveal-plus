import numpy as np
from numpy import save
from numpy import load
import sys
import pickle


filepath = sys.argv[1]
beta = sys.argv[2]
fileFull=sys.argv[3]
fileSeed=sys.argv[4]
conf=int(sys.argv[5])


print("running with flag ", conf)

if( len(sys.argv)<5):
    print("1- non labeled file \n 2- beta value \n 3-file to create the dictionary with full set of records \n 4-seed input file \n 5 -load seed 1")


positivos=0
negativos=0
labelSet=[]
#flag indicate to load file 
if( not conf):
    trainingSet=load('data.npy')
    print("file load")
    #print(m)
else:
    trainingSet=[]#np.zeros((1,10))
    print("matrix created")

#flag indicate to load file 
if( not conf):
    labelSet=load('labelset.npy')
    print("file load")
    #print(m)
else:
    labelSet=[]#np.zeros((1,10))
    print("label created")


d=[] 
#flag indicate to load file
if ( not conf):
    for i in range(50):
        pkl_file = open('myfile.pkl'+str(i), 'rb')
        d.append(pickle.load(pkl_file))
        pkl_file.close()
    #print(d)
    print("load dictionry ")
else:
    print("dictionry created")
    for i in range(50):
        d.append(dict())
        
        
def create_dictionary(filepath):

    vet=[0]*50
    with open(filepath) as fp:
        lines = fp.readlines()   
        for line in lines:
       #print("Line {}".format( line.strip()))
            
            att=line.split(" ")
            if(len(att)<2):
                continue
       #print(att[0], "+",att[2] )
            count=0
            for i in range(50):
                if "=" in att[i]:
                    value=att[i].split("=")[1]
                    if value == "0-1.0E-6":
                        continue
                    dictionary=d[i]
                    if(len(value)>0 and not dictionary.get(value)):                
                            
                        dictionary[value]=vet[i]
                        vet[i]+=1                     
    print(d)
    
def load_matrix_seed(filepath,trainingSet,labelSet):
    #print(m)
    with open(filepath) as fp:
        lines = fp.readlines()                    
        for line in lines:
            att=line.split(" ")
            if(len(att)<2):
                continue
            classe=0
            vetemp=[]
            for i in range(12):
                    
                if "=" in att[i]:
                    value=att[i].split("=")[1]
                    if value == "0-1.0E-6":
                        continue
                    if("CLASS" in att[i]):
                        classe=int(value)
                        continue
                    dictionary=d[i]
                    Mvalue=dictionary.get(value)
                    
                    vetemp.append(Mvalue)                   
            if (trainingSet==[]):
                trainingSet=np.array([vetemp])
                labelSet.append(classe)
            else:
                trainingSet=np.append(trainingSet, [vetemp], axis=0)
                labelSet=np.append (labelSet, classe)
            
    print(trainingSet.shape)
    return (trainingSet,labelSet)



def find_record_min(trainingSet,labelSet):
    global positivos
    global negativos
    print("find record min ")
    loop=True
    lineNumber=[]
    storeUnlabeledClass=[]
    with open(filepath) as fp:
        lines = fp.readlines()         
       # print("unlabeled dataset is ", lines)
        while loop:    
            totalRuleNumber=[]
            
            for line in lines:               
                if(len(lines)==0):
                    loop=False
                    break
                att=line.split(" ")
                vetRecord=[]                
                if(len(att)<2):
                    continue                    
                classe=0
                ruleNumber=0
                for i in range(12):                
                    if "=" in att[i]:
                        value=att[i].split("=")[1]
                        if value == "0-1.0E-6":
                            continue
                        if("CLASS" in att[i]):
                            classe=int(value)
                            continue    
                        dictionary=d[i]
                        Colvalue=dictionary.get(value)
                        vetRecord.append(Colvalue)
                            
               
                menor=0
                storeI=0                
                storeUnlabeledClass.append(classe)
                for i in range(0,len(trainingSet)):
                    if(labelSet[i]==0):
                        ruleNumber+=2**(np.count_nonzero(trainingSet[i].astype(int)==np.array(vetRecord)))*5
                    else:
                        ruleNumber+=2**(np.count_nonzero(trainingSet[i].astype(int)==np.array(vetRecord)))
                        #print("increase line number", ruleNumber)
                #print(vetRecord, " ",line)                  
                totalRuleNumber.append(ruleNumber)
                #salva  a linha adicionada
            print("training set ", trainingSet, " \n label set ", labelSet, " rule number ",totalRuleNumber, " unlabeled set size " , len(lines))
            minimo=totalRuleNumber.index(min(totalRuleNumber))
           #print("total de rules", totalRuleNumber, "minimo ", minimo, " ", len (lines), " lines", m[minimo], " ", lines[minimo])
            if minimo in lineNumber and storeUnlabeledClass[minimo]==0:
                print("new record is min ", minimo, " stopping active learning" ,lineNumber, "total positivos ", positivos, " negativos ", negativos)
                loop=False                    
            else:                
                lineNumber.append(minimo)               
                print("add record ",minimo, " records already inserted", lineNumber, " ", lines[minimo].split(" ")[1])
                #print ("new record is ",minimo," ",lines[minimo]) 
                trainingSet,labelSet=add_record_min(lines[minimo],trainingSet,labelSet) 
                print("removendo linha ", lines.pop(minimo), " ", len(lines))
                if(len(lines)==0):
                    loop=False
                    print("new record is min ", minimo, " stopping active learning" ,lineNumber, "total positivos ", positivos, " negativos ", negativos)
                    break


                
                
    return(trainingSet,labelSet)         
    
  


def add_record_min(r,trainingSet,labelSet):
    global positivos
    global negativos
    #print("adding record to the training set ", r.split(" ")[0])
    att=r.split(" ")
    classe=0
    vetemp=[]
    for i in range(12):
                    
        if "=" in att[i]:
            value=att[i].split("=")[1]
            if value == "0-1.0E-6":
                continue
            if("CLASS" in att[i]):
                classe=int(value)
                continue
                             
            if(classe==1):
                print("element not inserted because only one  relevant is present in the training st")
                break;
            dictionary=d[i]
            Mvalue=dictionary.get(value)
            vetemp.append(Mvalue)
              
    if(classe):
        positivos+=1
    else:
        negativos+=1
        trainingSet=np.append(trainingSet, [vetemp], axis=0)
        labelSet=np.append(labelSet,[classe], axis=0)   

   # print(i, "+",m, end=' ')
   # print()
    return(trainingSet,labelSet)


if (conf):
    create_dictionary(fileFull)
    trainingSet, labelSet=load_matrix_seed(fileSeed,trainingSet,labelSet)
    
trainingSet,labelSet=find_record_min(trainingSet,labelSet)
#add_record_min(record)




#save dictionary
for i in range(50):
    output = open('myfile.pkl'+str(i), 'wb')
    pickle.dump(d[i], output)
    output.close()


#save matrix
#print("print ----------",trainingSet)
save('data.npy', trainingSet)
save('labelset.npy', labelSet)


