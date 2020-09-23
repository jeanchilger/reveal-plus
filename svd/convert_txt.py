import sys
import csv


input_file = sys.argv[1]
output_file = sys.argv[2]
ground_file= sys.argv[3]

#d = int( sys.argv[3] )
#assert ( d > 0 )

reader = csv.reader( open( input_file ), delimiter = " " )

f=open( output_file, 'w' )
#writer = csv.writer(open( output_file, 'w' ), delimiter =",",quoting=csv.QUOTE_MINIMAL)


features=51

maior=0
min=[1]*features
max=[0]*features

for line in reader:
    i=0
    for at in line: 
        
        if ":" in at:
            #print (at)    
            string=''.join(at)
        #line=line.replace("\n","")
            
            modulos=string.split(":")[1]
            
            if(max[i]<float(modulos)):
                max[i]=float(modulos)
                #if(i==0):
                #    print(modulos)
            if(min[i]>float(modulos)):
                min[i]=float(modulos)
            i+=1
        #   print (i)
#print (min, "\n\n ",max)
d=features

reader = csv.reader( open( input_file ), delimiter = " " )
for line in reader:
	#print(line)
	label = line.pop( 0 )
#	print (label)	
	#if label in open(ground_file).read():
	if label == '1':
            label = 1
            ##print("achou arquivo certo")
	else:
            label = '0'
	if line[-1].strip() == '':
		line.pop( -1 )
                
	
	#print (line)	
	line = map(lambda x: tuple( x.split( ":" )), line)
	#print (line)	
	# ('1', '0.194035105364'), ('2', '0.186042408882'), ('3', '-0.148706067206'), ...
	pos=0
	new_line =  [ 0 ] * (d+1)
	
	
	try:
            for i, v in line:
                    i = int( i )
                    if i < d:
                            v=float(v)
                            if max[pos]==min[pos]:
                                temp=max[pos]
                            else:    
                                temp= (v-min[pos])/(max[pos]-min[pos])
                            #print (temp)
                            new_line[i] = round(temp,2)	
                    pos+=1                
	except:
            #print ("Error linha 80 ", line)
            #for i in line:
                #print (i)
            continue;
        
	new_line[d]=label
        
	#writer.writerow(new_line)
	for i in range(features-1):
                strin=str(new_line[i])+", "
                f.write(strin)
	f.write(str(new_line[features]) +"\n")
        
f.close()
