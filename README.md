# REVEAL (Refactor in progress)

High recall Information REtrieval (HIRE) aims at identifying only and (almost) all relevant documents for a given query. HIRE is paramount in applications such as systematic literature review, medicine, legal jurisprudence, etc. Supervised (classification) approaches are traditionally used in HIRE to produce a rank of relevant documents. However, such strategies depend on informative (very) large training sets with a wide variety of patterns to identify the relevant documents. In this context, active learning methods have proven to be quite useful to determine informative and non-redundant documents to compose these training sets, while reducing user effort for manual labeling. In this paper, we propose
REVEAL-HIRE – a new active framework for the HIRE task that selects a very reduced set of documents to be labeled, significantly mitigating the user’s effort. REVEAL-HIRE selects the most representative documents by following three steps. In the first one, an incremental supervised approach is used to produce a partial ranking based on the user query. In the second step, a new active learning strategy, called REVEAL, is applied to the top-ranked documents to choose the most informative ones to be labeled. A penalization factor is employed to select just the potentially relevant ones in such a sample, hard work due to the skewed nature of the task. This process is repeated until a stopping point is achieved in the third step: the final screening. This last step, which incorporates a newly proposed stopping heuristics, is designed to retrieve the last
remaining relevant documents with reduced effort. Experimental results demonstrate that our approach can achieve very high recall levels while reducing the labeled effort in up to 3.8 times when compared to the results obtained with state-of-the-art baselines, ultimately producing the best trade-off between recall and labeling effort.

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

