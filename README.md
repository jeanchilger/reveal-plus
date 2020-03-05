# Sampled Continuous Active Learning
Sampled Continuous Active Learning (SCAL) applied to medical articles

## Installation

1. `git clone https://github.com/HTAustin/CAL.git`
2. Intall Sofia-ML package: https://code.google.com/archive/p/sofia-ml/
3. Make the kissdb indexer. `cd CAL && make`
4. Change the path for Sofia-ML in doAll_Baseline
```
SOFIA="/the/path/to/sofia-ml-read-only/src/sofia-ml"
```

## Usage

### Running a single time
1. Apply 4-gram tf-idf features: `bash doAll_Baseline_4gram`;
2. The output of BMI are stored in `result/` folder;
3. The gain curve can be plotted by analyzing `$TOPIC.record.list`.

### Running several times (taking means and standard deviation over executions)
This option will run the code several times and take means and standard deviations over all executions' result, given a more solid and reliable output (although it takes longer).

1. Change the `SAMPLES` variable in `wrap_results` to the number of times you wish to run the code;
2. Select specific topics to be evaluated by adding them to the `TOPICS_CONSIDERED` list in the same file;
3. Run `bash wrap_results`;
4. The results are stored under `results/` folder, separated by topic. Within each topic folder, there are results of individual runs (`rel.{exec_number}.rate.csv`) and a file with means and standard deviations of them all (`rel.general.rate.csv`).

## Credits

This is a work intended to improve the CAL (Continuous Active Learning) protocol for TAR (Technology Assisted Review) processes by CORMACK, Gordon V. and GROSSMAN, Maura R. (2014). The code used is from [HTAustin](https://github.com/HTAustin)'s [CAL repository](https://github.com/HTAustin/CAL)

