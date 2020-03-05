import os
import sys
import csv
import statistics

ALL_RESULT_HELPER = sys.argv[1]

if not os.path.exists("results"):
    os.mkdir("results")

with open(ALL_RESULT_HELPER, "r") as runs_describer:
    for _line in runs_describer:

        # topic execution_number result_file
        line = _line.split()

        topic_dir = "results/" + line[0]
        topic_file = topic_dir + "/rel." + line[1] + ".rate.csv"

        if not os.path.exists(topic_dir):
            os.mkdir(topic_dir)

        # writes positives and all to a csv file
        with open(line[2], "r") as rel_file, open(topic_file, "w") as rel_dst_file:
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

# calculates mean and standard deviation over data samples taken

results_path = "results/"
for topic_dir in os.listdir(results_path):

    topic_path = os.path.join(results_path, topic_dir)
    rel_general_file = os.path.join(topic_path, "rel.general.rate.csv")

    topic_rel_files = [os.path.join(topic_path, f) for f in os.listdir(topic_path) if os.path.isfile(os.path.join(topic_path, f))]

    positives = []
    recall = []
    all_docs = []
    limited_docs = []

    result_count = 0

    for _topic_file in topic_rel_files:
        with open(_topic_file, "r") as topic_file:
            topic_reader = csv.reader(topic_file)

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
