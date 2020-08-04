import os
import sys

#if os.path.exists(sys.argv[1]):
    #print("file input not exists")
    #exit(1)

def main( topic):
    mapping = {}

    

    svmfil = {}
    with open("clef2018.svm.fil", "r") as mapping_file:
        for line in mapping_file.readlines():
            info = line.strip().split(" ",1)
            
            svmfil[info[0]] = info[1]
            

    
    with open("data/"+topic+".pids", "r") as mapping_file,  open(topic+".svm.fil", "w") as outdocs:
        for line in mapping_file.readlines():
            info = line.strip().split(" ")
            #print (info[1], svmfil[info[1]])
            try:
                outdocs.write(info[1]+ " "+ svmfil[info[1]]+"\n")
            except:
                print("erro label ", line)
                pass
    
    #for i in mapping[1]:
        #print (i)
        #print(svmfil[i])
def usage(args):
    print("Usage: {0} <topic>".format(args[0]))




if __name__ == "__main__":
    filename = None

    if len(sys.argv) >= 1:        
        topic = sys.argv[1]         
        main(topic)
    else:
        usage(sys.argv)
        exit(1)
