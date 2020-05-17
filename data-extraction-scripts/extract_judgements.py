import os

if os.path.exists("qrels.cord_base.list"):
    os.remove("qrels.cord_base.list")

with open("qrels-rnd1.txt", "r") as qrels_read, open("qrels.cord_base.list", "a") as qrels_write:
    while True:
        line = qrels_read.readline()

        if not line:
            break

        info = line.strip().split(' ')
        info.remove('')

        if info[3] == "2":
            topic_id = "tr" + info[0]
            dummy = info[1]
            cord_id = info[2]
            judgment = "1"

            qrels_write.write(" ".join([topic_id, dummy, cord_id, judgment]) + "\n")



