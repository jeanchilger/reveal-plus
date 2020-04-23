# Scalable Continuous Active Learning (Refactor in progress)
Scalable Continuous Active Learning (SCAL) applied to medical articles

**Coordinating professor:** Guilherme Dal Bianco ([github](https://github.com/dbguilherme), [lattes]( http://lattes.cnpq.br/5152594034228273))

**Participants:** [Emili Willinghoefer](https://github.com/Emiliwillinghoefer), [Jean Carlo Hilger](https://github.com/jeanchilger) and [Matheus Vinícius Todescato](https://github.com/mvtodescato)

**NOTE:** Previously work were made in [mvtodescato's CAL fork](https://github.com/mvtodescato/CAL), however, since we don't intend to merge it with the original upstream, we decided to create another repository, therefore all modifications since 5 March, 2020 will be made in this repository.

## Installation

1. Clone the repository: `git clone https://github.com/JeanCHilger/SCAL.git`
2. Make kissdb indexer (from inside repository folder): `make`

Ready to use

## Usage

We are working only with the `4-gram` version.

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

