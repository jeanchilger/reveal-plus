import sys

f1 = open ( sys.argv[1] , 'r')
f2 = open ( "flagOut" , 'w')

calculado=float(sys.argv[2])

for line in f1:
    if float(line.split(' ')[1]) >= calculado:
        f2.write(line.split(' ')[0])
        break
    
f2.close()
f1.close()