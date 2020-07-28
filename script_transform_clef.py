import os
import sys

#if os.path.exists(sys.argv[1]):
    #print("file input not exists")
    #exit(1)

def main(filename, topic, filemapping):
    mapping = {}

    with open(filemapping, "r") as mapping_file:
        for line in mapping_file.readlines():
            info = line.strip().split(" ")
            mapping[info[1]] = info[0]


    count=0
    with open(filename, "r") as ordedocs,  open("clefevaluation", "a") as outdocs:
        for line in ordedocs.readlines():
            info = line.strip()
            outdocs.write(topic+" NF "+mapping[info]+ " "+ str(count) +" "+ str(-count)+" rev \n")
            count+=1



def usage(args):
    print("Usage: {0} <fileinput> <topic>".format(args[0]))




if __name__ == "__main__":
    filename = None

    if len(sys.argv) >= 3:
        filename = (sys.argv[1])
        topic = sys.argv[2] 
        filemapping= sys.argv[3] 
        main(filename,topic, filemapping)
    else:
        usage(sys.argv)
        exit(1)
