"""
Generates a file with the judgements, that is,
the relevant documents at judgement/qrels.cord-19.list.

Input:
    relevance_file => cord-19 file with relevance judgements (e.g. qrels-rnd1.txt).
"""

import os
import sys

relevance_file = sys.argv[1]

if os.path.exists("judgement/qrels.cord-19.list"):
    os.remove("judgement/qrels.cord-19.list")

mapping = {}
with open("map", "r") as mapping_file:
    for line in mapping_file.readlines():
        info = line.strip().split(",")
        mapping[info[0]] = info[1]

with open(relevance_file, "r") as qrels_read, open("judgement/qrels.cord-19.list", "a") as qrels_write:
    while True:
        line = qrels_read.readline()

        if not line:
            break

        info = line.strip().split(" ")
        info.remove("")

        if info[3] == "2":
            topic_id = "tr" + info[0]
            dummy = info[1]
            cord_id = mapping[info[2]]
            judgement = "1"

            qrels_write.write(" ".join([topic_id, dummy, cord_id, judgement]) + "\n")
