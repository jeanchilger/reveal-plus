"""
Generates the file with topic:query pairs,
at judgement/cord-19.topic.stemming.txt.

Input:
    queries_file => cord-19 file with topics description (e.g. topics-rnd1.xml).
"""

import os
import sys
import xml.etree.ElementTree as ET

queries_file = sys.argv[1]

tree = ET.parse(queries_file)
root = tree.getroot()

if os.path.exists("judgement/cord-19.topic.stemming.txt"):
    os.remove("judgement/cord-19.topic.stemming.txt")

with open("judgement/cord-19.topic.stemming.txt", "a") as cord_base_file:
    for elem in root:
        query = elem[0].text
        topic = "tr" + elem.attrib["number"]

        cord_base_file.write(topic + ":" + query + "\n")
