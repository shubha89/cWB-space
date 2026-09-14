# Sangria candidate plot

`python examples/plot_sangria.py` recreates Figure 9 of arXiv:2609.12986v1. Outputs default to `outputs/`; use `--output-dir DIRECTORY` to change this.

`candidates.csv` is the complete 155-row saved candidate catalogue, preserved verbatim. `associations.csv` contains the six recorded MBHB associations and source/observed slice hashes. The script checks candidate identity, rank, time and event statistic before highlighting the six entries. The other 149 candidates remain displayed. Rank labels are ranking positions, not catalogue source numbers.

Times are days from the supplied observation start; the vertical axis is the saved event statistic on a logarithmic display. The plot and these CSV files do not constitute a new search, a calibrated significance estimate or a population recovery-efficiency measurement. They contain no observed or injected waveform time series; reproducing the full-year search requires the original observation and original search configuration separately.
