import os
import sys
import csv
import statistics

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

        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        # writes positives and all to a csv file
        with open(os.path.join(src_dir, "rel.rate"), "r") as rel_file, open(scal_file, "w") as rel_dst_file:
            rel_writer = csv.writer(rel_dst_file)

            rel_writer.writerow(["positives", "all_docs", "limited_docs", "recall"])

            all_so_far = 0
            limited_so_far = 0
            total_positives = int(rel_file.readlines()[-1].split()[4])

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



# calculates mean and standard deviation over data samples taken
results_path = "results/"
for dst_dir in os.listdir(results_path):

    topic_path = os.path.join(results_path, dst_dir)
    rel_general_file = os.path.join(topic_path, "rel.general.rate.csv")

    topic_rel_files = [os.path.join(topic_path, f) for f in os.listdir(topic_path) if f.split(".")[0] == "rel"]
    print("AAAA", len(topic_rel_files))

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

    if result_count > 1:
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
    else:
        print("There was only one run, impossible calculate stdevs and means.")
