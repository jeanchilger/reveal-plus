import os

if os.path.exists("judgement/qrels.cord-19.list"):
    os.remove("judgement/qrels.cord-19.list")

mapping = {}
with open("html-pid-mapping", "r") as mapping_file:
    for line in mapping_file.readlines():
        info = line.strip().split(" ")
        mapping[info[0]] = info[1]

with open("full.test.content.2018.qrels", "r") as qrels_read, open("../judgement/qrels.clef.list", "a") as qrels_write:
    while True:
        line = qrels_read.readline()
        #print(line)
        if not line:
            break

        info = line.strip().split(" ")
        
        print(info[3])
        if info[3] == "1":
            topic_id = info[0]
            dummy = info[1]
            cord_id = mapping[info[2]]
            judgement = "1"

            qrels_write.write(" ".join([topic_id, dummy, cord_id, judgement]) + "\n")



