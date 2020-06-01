"""
Counts how many relevant documents exists per topic.

Based on judgement/qrels.cord-19.list
"""

import os

topic_count = {}

with open("../judgement/qrels.cord-19.list", "r") as qrels_file:

    for _line in qrels_file.readlines():
        line = _line.split()

        if line[0] in topic_count.keys():
            topic_count[line[0]] += 1

        else:
            topic_count[line[0]] = 1


# Generate the answer
with open("relevant.topic.counting", "w") as result_file, open("../judgement/cord-19.topic.stemming.txt", "r") as query_file:
    for key in topic_count.keys():
        result_file.write(key + ":" + query_file.readline().split(":")[1].strip() + ":" + str(topic_count[key]) + "\n")


