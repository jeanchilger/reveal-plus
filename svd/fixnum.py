import sys
import numpy as np

f1 = open ( sys.argv[1] , 'r')

for linha in f1:
    splitLinha=linha.split(' ',1)


    print((((splitLinha[0])).zfill(15)) , '' ,(splitLinha[1]), end='')




#000000000000001



