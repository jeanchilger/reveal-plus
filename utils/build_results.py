import csv
import os
import shutil
import statistics
import sys

ABS_PATH = sys.argv[1]
print(ABS_PATH)

ALL_RESULTS_HELPER="all_results/runs.describe.txt"

if not os.path.exists("results"):
    os.mkdir("results")

with open(ALL_RESULTS_HELPER, "r") as runs_describer:
    for _line in runs_describer:

        # topic execution_number result_file
        line = _line.split()

        dst_dir = "results/" + line[0]
        src_dir = "all_results/result-" + line[1]

        scal_file = dst_dir + "/rel." + line[1] + ".rate.csv"
        reveal_file = dst_dir + "/reveal." + line[1] + ".final.csv"

        total_positives = int(line[2])

        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        # writes positives and all to a csv file
        with open(os.path.join(src_dir, "rel.rate"), "r") as rel_file, open(scal_file, "w") as rel_dst_file:
            rel_writer = csv.writer(rel_dst_file)

            rel_writer.writerow(["positives", "all_docs", "limited_docs", "recall"])

            all_so_far = 0
            limited_so_far = 0

            if total_positives == 0:
                total_positives = 1

            rel_file.seek(0)

            for _rel_line in rel_file:
                rel_line = _rel_line.split()

                positives = int(rel_line[4])
                all_docs = int(rel_line[1]) + all_so_far
                limited_docs = int(rel_line[2]) + limited_so_far

                all_so_far = all_docs
                limited_so_far = limited_docs

                rel_writer.writerow([positives, all_docs, limited_docs, "{:.5f}".format(positives/total_positives)])

        # saves final reveal to a csv file
        with open(os.path.join(src_dir, "reveal.final")) as rev_file, open(reveal_file, "w") as rev_dst_file:

            rev_writer = csv.writer(rev_dst_file)

            rev_writer.writerow(["recall", "precision", "lab_eff"])

            for _rev_line in rev_file:

                rev_line = _rev_line.split()

                recall = float(rev_line[0])
                precision = float(rev_line[1])
                lab_eff = int(rev_line[2])

                rev_writer.writerow(["{:.5f}".format(recall), "{:.5f}".format(precision), lab_eff])

        # copy logs
        shutil.copy(src_dir + "/runs.log", dst_dir + "/run." + line[1] + ".log")

results_path = "results/"

# calculates mean and standard deviation over data samples taken
# *SCAL data
for dst_dir in os.listdir(results_path):

    topic_path = os.path.join(results_path, dst_dir)
    rel_general_file = os.path.join(topic_path, "rel.general.rate.csv")

    topic_rel_files = [os.path.join(topic_path, f) for f in os.listdir(topic_path) if f.split(".")[0] == "rel"]

    # just one run (or some mistake happened)
    if len(topic_rel_files) <= 1:
        print("There was only one run, impossible calculate stdevs and means.")
        break

    positives = []
    recall = []
    all_docs = []
    limited_docs = []

    result_count = 0

    for _scal_file in topic_rel_files:
        with open(_scal_file, "r") as scal_file:
            topic_reader = csv.reader(scal_file)

            curr_idx = -1

            for row in topic_reader:
                # skip header
                if curr_idx == -1:
                    curr_idx += 1
                    continue

                if result_count == 0:
                    positives.append([])
                    recall.append([])
                    all_docs.append(int(row[1]))
                    limited_docs.append(int(row[2]))

                positives[curr_idx].append(float(row[0]))
                recall[curr_idx].append(float(row[3]))

                curr_idx += 1

        result_count += 1

    with open(rel_general_file, "w") as rel_gen:

        rel_gen_writer = csv.writer(rel_gen)

        rel_gen_writer.writerow(["all_docs", "limited_docs", "positives_mean", "positives_stdev", "recall_mean", "recall_stdev"])

        for i in range(len(positives)):
            rel_gen_writer.writerow([
                all_docs[i],
                limited_docs[i],
                "{:.5f}".format(statistics.mean(positives[i])),
                "{:.5f}".format(statistics.stdev(positives[i])),
                "{:.5f}".format(statistics.mean(recall[i])),
                "{:.5f}".format(statistics.stdev(recall[i]))
            ])

# calculates mean and standard deviation over data samples taken
# *REVEAL data
for dst_dir in os.listdir(results_path):

    topic_path = os.path.join(results_path, dst_dir)
    rev_general_file = os.path.join(topic_path, "reveal.general.final.csv")

    topic_rev_files = [os.path.join(topic_path, f) for f in os.listdir(topic_path) if f.split(".")[0] == "reveal"]

    # just one run (or some mistake happened)
    if len(topic_rev_files) <= 1:
        break

    recall = []
    precision = []
    lab_eff = []

    result_count = 0

    for _reveal_file in topic_rev_files:
        with open(_reveal_file, "r") as reveal_file:
            rev_reader = csv.reader(reveal_file)

            curr_idx = -1

            for row in rev_reader:
                # skip header
                if curr_idx == -1:
                    curr_idx += 1
                    continue

                recall.append(float(row[0]))
                precision.append(float(row[1]))
                lab_eff.append(int(row[2]))

    with open(rev_general_file, "w") as reveal_gen:
        rev_gen_writer = csv.writer(reveal_gen)

        rev_gen_writer.writerow([
            "recall_mean",
            "recall_stdev",
            "precision_mean",
            "precision_stdev",
            "lab_eff_mean",
            "lab_eff_stdev"
        ])

        rev_gen_writer.writerow([
            "{:.5f}".format(statistics.mean(recall)),
            "{:.5f}".format(statistics.stdev(recall)),
            "{:.5f}".format(statistics.mean(precision)),
            "{:.5f}".format(statistics.stdev(precision)),
            "{:.5f}".format(statistics.mean(lab_eff)),
            "{:.5f}".format(statistics.stdev(lab_eff))
        ])
