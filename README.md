[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-red.svg)](https://www.gnu.org/licenses/gpl-3.0)

# REVEAL-plus

## Overview
High-Recall Information Retrieval (HRIR) - the identification of nearly all relevant documents within a set of them, given a search query - is a pivotal task in a wide range of applications [1] such as electronic discovery and systematic review. The relevant documents are judged this way by the user conducting the search, which may cause excessive effort (for the user) in classifying documents.

For the given context, several researches have focused on reducing this effort, still providing a high recall. As a result a considerable number of techniques have emerged, among which it is worth mentioning REVEAL - RelEVant rulE-based Active Learning (check out [1] for more details) - which uses active learning and association rules to improve the HRIR performance.

This project intend to improve the REVEAL by providing:
- An enhanced starting point for the method; and
- A clear and definite sopping criteria;

## Installation
1. Clone this repository and enter it:
`git clone https://github.com/jeanchilger/scal.git && cd SCAL`
1. Inside repository, build kissdb indexer and other necessary binaries:
`make`

You are ready to go

## Usage
The bash file named `main` is the entry point for the system. Type `./main <option>` for using it. Check the available options below.

```bash
-s <samples>, --samples=<samples>
      Set <samples> as the quantity of executions that will occurs.
      After, the mean and standard deviation over the samples are taken.
      The standard value is 1.

-c <corpus>, --corpus=<corpus>
      Specifies the name of the corpus to be used.

-t <topic-list>, --topics=<topic-list>
      Specifies which topics will be computed by the method.
      <topic-list> must be a space separated string, containing
      one or more topics.

-v, --verbose
      If specified, verbose messages will be shown during execution.

-o, --off-colors
      Turns off colors of terminal outputs.

-h, --help
      Show a message like this.
```

## Datasets
Some dataset we've adapted and used for assessment are listed below.

### CLEF 2017 Dataset
CLEF 2017 development set, which was created based on the Diagnostic Test Accuracy (DTA) systematic reviews conducted by the [Cochrane Library](https://www.cochranelibrary.com/).

*See [3].*

### CORD-19
[COVID-19 Open Research Dataset](https://www.semanticscholar.org/cord19).

*See [4].*

## Team
### Coordinating professor
- Guilherme Dal Bianco
  - [GitHub](https://github.com/dbguilherme)
  - [Lattes (pt-br Curriculum)](http://lattes.cnpq.br/5152594034228273)

### Academics participating
- Emili Willinghoefer (early work)
  - [GitHub](https://github.com/Emiliwillinghoefer)
- Jean Carlo Hilger
  - [GitHub](https://github.com/jeanchilger)
- Matheus Vinícius Todescato
  - [GitHub](https://github.com/mvtodescato)

## Credits
The code used as basis is from [HTAustin](https://github.com/HTAustin)'s [CAL repository](https://github.com/HTAustin/CAL), we adapted his code from CAL to SCAL and later to the REVEAL method.

## References
Core articles that guided this project.
- [1] Guilherme Dal Bianco. **Reveal-hire - a new active framework for the high recall task**. In *Proceedings of ACM Conference (Conference’17)*, 2018.
- [2] Gordon  V.  Cormack  and  Maura  R.  Grossman. **Scalability  of  continuous active learning for reliable high-recall text classification**. In *Proceedings of the 25th ACM International Conference on Information and Knowledge Management*, 2016.
- [3] Evangelos Kanoulas, Dan Li, Leif Azzopardi, and Rene Spijker. **CLEF 2017 technologically assisted reviews in empirical medicine overview**. *CEUR Workshop Proceedings*, 2017.
- [4] Ellen M. Voorhees, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman, William R. Hersh, Kyle Lo, Kirk Roberts, Ian Soboroff, and Lucy Lu Wang. **Trec-covid: Constructing a pandemic information retrieval test collection**. *ArXiv*, 2020.
