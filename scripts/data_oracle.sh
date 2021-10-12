#!/bin/bash

seed=123456789
#ticker=IBM
#dates=(20190628 20190627 20190626 20190625 20190624
#       20190621 20190620 20190619 20190618 20190617
#       20190614 20190613 20190612 20190611 20190610
#       20190607 20190606 20190605 20190604 20190603)

ticker=AAPL
dates=(20181221)
for d in ${dates[*]}
  do
    python -u abides.py -c data_oracle  -t $ticker -d $d -s $seed -l data_oracle_${ticker}_${d}
  done


cd util/plotting && python -u liquidity_telemetry.py ../../log/data_oracle_AAPL_20181221/EXCHANGE_AGENT.bz2 ../../log/data_oracle_AAPL_20181221/ORDERBOOK_AAPL_FULL.bz2 \
 -o data_oracle_AAPL.png -c configs/plot_09.30_11.30.json && cd ../../