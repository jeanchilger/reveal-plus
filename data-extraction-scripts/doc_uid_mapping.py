"""
Generates a mapping file named topic.id.mapping
with two space separated columns.

The first column is the document uid in cord-19
base, and the second column is the document name
to be used in the SCAL format.
"""

import pandas
import os

data = pandas.read_csv("metadata.csv")

uids = data.cord_uid.tolist()

print("Total columns:", len(uids))
print("Unique uids:", len(set(uids)))

repeated = []      # list of repeated (used just for information)

unique_count = 0   # tracks document name
doc_mapping = {}   # maps names (could be used for uniqueness check)

if os.path.exists("topic.id.mapping"):
    os.remove("topic.id.mapping")

with open("topic.id.mapping", "a") as map_file:
    for elem in uids:
        if elem in doc_mapping.keys():
            repeated.append(elem)

        else:
            doc_mapping[elem] = str(unique_count).zfill(7)
            unique_count += 1

            map_file.write(" ".join([elem, doc_mapping[elem]]) + "\n")

        doc_name = doc_mapping[elem]

print("Repeated elemets found:", len(repeated))
print("Unique repeated elements:", len(set(repeated)))
