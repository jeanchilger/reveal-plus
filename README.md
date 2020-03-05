# BMI local Implementation

## Installation

1. `git clone https://github.com/HTAustin/CAL.git`
2. Intall Sofia-ML package: https://code.google.com/archive/p/sofia-ml/
3. Make the kissdb indexer. `cd CAL && make`
4. Change the path for Sofia-ML in doAll_Baseline
```
SOFIA="/the/path/to/sofia-ml-read-only/src/sofia-ml"
```

## Usage

1. Apply word tf-idf features: `bash doAll_Baseline`
2. Or apply 4-gram tf-idf features: `bash doAll_Baseline_4gram`
3. The output of BMI are stored in `result/` folder. 
4. The gain curve can be plotted by analyzing `$TOPIC.record.list`
5. Change number of threads in `doAll_Baseline` by changing the variable `MAXTHREADS` (default=4)


## Contribute

Please feel free to open issues and report bugs.

## License

[![GNU GPL v3.0](http://www.gnu.org/graphics/gplv3-127x51.png)](http://www.gnu.org/licenses/gpl.html)
