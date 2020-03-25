#import collections
#import sys

#dataset="medline"
#print(sys.argv[1])
#file = open(sys.argv[1], "r")

#out= open(sys.argv[2], "w")
##ground= open("/home/guilherme/Downloads/l2knng/build/data/" + dataset+"Ground.csr", "w")
#lineNumber=0
#colunaNumber=0
#elementsNumber=0
#memoria="0"
#set_data=[]

##00001 291:0.02701503 300:0.05445237 301:0.06792670 478:0.03020137 491:0.08438419 509:0.05445237 526:0.02271182 536:0.04711743 663:0.01640210 683:0.02126500 840:0.03576732 850:0.05596973 890:0.07502736 894
##:0.06768917 946:0.04048635 954:0.11634092 980:0.04406309 1051:0.05452759 1121:0.15351788 1170:0.04145513 1266:0.18805912 1272:0.24272439 1481:0.03335054 1485:0.13044648 1518:0.02510089 1876:0.00275196 192
##5:0.05998979 2040:0.05828959 2311:0.14729737 2318:0.05613969 2325:0.01330839 2373:0.09157013 2540:0.10129615 2739:0.05673878 3196:0.19631219 3286:0.07374378 3313:0.03662727 3484:0.16708679 3531:0.03257208
 ##3561:0.04440346 3696:0.09992474 3787:0.13810099 3818:0.08852931 3886:0.06244799 3897:0.06946143 3899:0.03020751 4041:0.00275427 4116:0.15223949 4203:0.06297021 4304:0.07576246 4594:0.04161022 4684:0.1076
##7378 4753:0.07441140 4762:0.07132483 4870:0.04089873 4871:0.03487033 5136:0.05527881 5332:0.06444359 5476:0.02556551 5486:0.07576246 5503:0.05729603 5608:0.09505276 5675:0.04645878 5679:0.07524767 5745:0.
##05037577 5802:0.06903577 5814:0.04696073 5863:0.07441140 5971:0.07008423 5999:0.02691525 6625:0.04291919 6662:0.04189993 6937:0.01630885 7005:0.05721915 7086:0.06663889 7114:0.03702262 7212:0.04209750 752
##7:0.08466594 7580:0.10657384 7586:0.07191658 7823:0.04200594 7828:0.01689450 7849:0.02690971 8224:0.02208897 8255:0.01206013 8343:0.03735116 8502:0.10981086 8510:0.05237395 8724:0.11282946 9020:0.02265325
 ##9023:0.04544989 9192:0.03638806 9261:0.05510129 9280:0.07359926 9324:0.09650971 9472:0.03735116 9606:0.03356676 9638:0.05428196 9661:0.03239648 9757:0.01975072 10007:0.05177936 10233:0.07773509 10237:0.1
##2087380 10423:0.03111120 10689:0.07336999 10805:0.04112413 10899:0.07191658 10980:0.05079460 11176:0.05726817 11212:0.01965964 11237:0.03450633 11361:0.05834283 11383:0.07550754 11405:0.01219678 11411:0.0
##2803449 11460:0.05866690 11497:0.02016946 11510:0.03550056 11528:0.06244531 11684:0.02769585 11910:0.03756560 12096:0.06293367 12117:0.06109799 12519:0.12221598 12590:0.05974985 12694:0.02373902 12734:0.0
##7369799 12794:0.05433421 12884:0.05632111 12980:0.06119411 13251:0.03625053 13379:0.05911195 13448:0.07176610 13593:0.06736510 13634:0.00159054 13647:0.08346185 13878:0.03717162 13916:0.08063961 14057:0.0
##1131371 14073:0.07563169 14094:0.08852931 14101:0.03978829 14347:0.05168306 14380:0.05974985 14410:0.03117638 14447:0.16968533 14614:0.08856065 14616:0.16708679 14661:0.09554509 14774:0.03586540 14902:0.1
##7310921 14981:0.07028858 15000:0.03099240 15183:0.06080163 15260:0.06146250 15280:0.05934031 15399:0.08612526 15698:0.08124513 16103:0.17704793 16158:0.00000000 16179:0.01646179 16185:0.01629764 16192:0.0
##3076602 16202:0.06667740 16215:0.03681537 16250:0.06931834 16265:0.06643864 16311:0.09496896 16332:0.00076483 16359:0.01745500 16408:0.06715137 16442:0.06135752 16488:0.02566912 16869:0.02970149 17144:0.1
##3809105 17380:0.05213868 17512:0.01662303 17666:0.06126197 17669:0.05963152 17704:0.06194592 17706:0.06999387 17777:0.01887519 17847:0.01419482 17905:0.02936330 18008:0.02119138 18118:0.02407956 18143:0.0
##7907326 18236:0.07253653




##map={}
#maior=0
#for line in file:
    #if ":" in line:
        
        
    ##line=line.replace("\n","")

        #modulos=line.split(":")[0]
        #if(maior<int(modulos)):
            #maior=modulos
    ##print(modulos)
    ##if ":" in modulos[0] and not "qid" in modulos[0]:
        ##print("errrooo ", modulos[0] , lineNumber)
        ##continue;
    ##if modulos[0] in map:
        ##list=map[modulos[0]]
        ##list.append(lineNumber)
        ##map[modulos[0]]=list
    ##else:
        ##list=[]
        ##list.append(lineNumber)
        ##map[modulos[0]]=list


    ##for i in range(1, len(modulos)):
        ##if(len(modulos[i])==0 or  len(modulos[i])==1 or  "qid" in modulos[i] or  "docid" in modulos[i] or  "G" in modulos[i]):
            ##continue;
        ##value=modulos[i].split(":")
       ### print(modulos,end=" ")
        ###print(value[0], value[1], sep=" ",end=" ")
        ##if(float(value[1])==0.0):
            ##continue
        ##if(int(value[0])>colunaNumber):
            ##colunaNumber=int(value[0])
        ###print (value[0])
       ### if value[0] not in set_data and ":" in modulos[i]:
        ##set_data.append(value[0]) 
            ##print (set_data)
           ## if(int(value[1])!=0):
           ##     elementsNumber+=1

    ##if(lineNumber>10):
    ##    break
##out.write(str(lineNumber) +" "+str(colunaNumber) + " "+ str(len(set_data))+  " \n")
#set_data=""
#print("segunda parte")
#file = open(sys.argv[1], "r")
#for line in file:
    #lineNumber+=1;
    #line=line.replace("\n","")

    #modulos=line.split(" ")
    
    #for i in range(1, len(modulos)):
        #if(len(modulos[i])<3 or len(modulos[i])==1 or  "qid" in modulos[i] or  "docid" in modulos[i] or  "G" in modulos[i]):
            #continue;
        #value=modulos[i].split(":")
        
        
        #out.write(str(value[1]))
        #if(float(value[0])!=64):
            #out.write(",")
            ##print(value[1])
    #out.write("\n")
    

##od = collections.OrderedDict(sorted(map.items()))

##for k, v in od.items():
    ###print (k)
    ##for w in (v):
        ###print(w)
        ##ground.write(str(w) + " 0.00 ")
    ##ground.write("\n")

##print("lineNumber ",lineNumber)
##print("colunaNumber ",colunaNumber)
##print("elementsNumber ",len(set_data))


"""
convert libsvm file to csv'
libsvm2csv.py <input file> <output file> <X dimensionality>
"""

import sys
import csv

input_file = sys.argv[1]
output_file = sys.argv[2]

#d = int( sys.argv[3] )
#assert ( d > 0 )

reader = csv.reader( open( input_file ), delimiter = " " )
writer = csv.writer( open( output_file, 'wb' ))


#map={}
maior=0
for line in reader:
    for at in line: 
        
        if ":" in at:
            #print (at)    
            str=''.join(at)
        #line=line.replace("\n","")
            
            modulos=str.split(":")[0]
            
            if(maior<int(modulos)):
                maior=int(modulos)

print (maior)
d=maior
reader = csv.reader( open( input_file ), delimiter = " " )
for line in reader:
	label = line.pop( 0 )
	if line[-1].strip() == '':
		line.pop( -1 )
		
	#print (line)
	
	line = map( lambda x: tuple( x.split( ":" )), line )
	#print (line)
	# ('1', '0.194035105364'), ('2', '0.186042408882'), ('3', '-0.148706067206'), ...
	
	new_line = [ label ] + [ 0 ] * d
	for i, v in line:
		i = int( i )
		if i <= d:
			new_line[i] = v		
	writer.writerow(new_line)
	#break;
