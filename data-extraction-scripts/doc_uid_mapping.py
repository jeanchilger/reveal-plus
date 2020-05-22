"""
Generates a mapping file named topic.id.mapping
with two space separated columns.

The first column is the document uid in cord-19
base, and the second column is the document name
to be used in the SCAL format.
"""

import os

repeated = []      # list of repeated (used just for information)

unique_count = 0   # tracks document name
doc_mapping = {}   # maps names (could be used for uniqueness check)

if os.path.exists("topic.id.mapping"):
    os.remove("topic.id.mapping")

with open("topic.id.mapping", "a") as map_file, open("docids-rnd1.txt", "r") as valid_uids_file:
    valid_uids = valid_uids_file.readlines()

    print("Total of elements:", len(valid_uids))
    print("Unique elements:", len(set(valid_uids)))

    for _uid in valid_uids:
        uid = _uid.strip()
        if uid in doc_mapping.keys():
            repeated.append(uid)

        else:
            doc_mapping[uid] = str(unique_count).zfill(7)
            unique_count += 1

            map_file.write(" ".join([uid, doc_mapping[uid]]) + "\n")

        doc_name = doc_mapping[uid]

print("Repeated elements found:", len(repeated))
print("Unique elements within the repeated:", len(set(repeated)))
