"""
Generates a mapping file named topic.id.mapping
with two space separated columns.

The first column is the document uid in cord-19
base, and the second column is the document name
to be used in the SCAL format.

Input:
    File with valid uids (e.g. docids-rnd1.txt).
"""

import pandas
import os
import sys

valid_docs_file = sys.argv[1]

repeated = []      # list of repeated (used just for information)
invalid = []       # list of invalid doc ids within metadata

data = pandas.read_csv("metadata.csv")

all_uids = [line.strip() for line in data.cord_uid.tolist()]

unique_count = 0   # tracks document name
doc_mapping = {}   # maps names (could be used for uniqueness check)

if os.path.exists("topic.id.mapping"):
    os.remove("topic.id.mapping")

with open("topic.id.mapping", "a") as map_file, open(valid_docs_file, "r") as valid_uids_file:
    valid_uids = [line.strip() for line in valid_uids_file.readlines()]

    for uid in valid_uids:

        if uid in all_uids:

            if uid in doc_mapping.keys():
                repeated.append(uid)

            else:
                doc_mapping[uid] = str(unique_count).zfill(7)
                unique_count += 1

                map_file.write(" ".join([uid, doc_mapping[uid]]) + "\n")

        else:
            invalid.append(uid)


print("Repeated (valid) elements found:", len(repeated))
print("Unique elements within the repeated:", len(set(repeated)))
print("Invalid docs:", len(invalid))
print("Valid docs:", len(doc_mapping.keys()))

print(invalid)
