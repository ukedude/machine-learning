# Machine Learning Engineer Nanodegree
## Specializations
## Project:  Capstone Project - Auto Trader

**Note**

"project report.pdf" contains the report on the project and results

"stock trader.ipynb" shows what analysis was done and examples of running the code.

trader.py is the main code module
data_utils.py contains code for data loading and deriving features
visuals.py contains code for plotting data

trader can be run from the command line:

trader.py -s <stocks> -i <investment> --start_train <starttraindate> --end_train <endtraindate> --start_test <starttestdat> --end_date <endtestdate> -f <features> -v
<stocks> is a list of trading symbols e.g. APPL or 'APPL,GE'
features to use - one or more of  ['MA(20)','UB(20)','LB(20)','MA(50)','UB(50)','LB(50)']

pandas_datareader is required for loading data. It can be installed using:

pip install pandas_datareader

yahoo_finance was used intially but found to be unreliable.  It may still be refernced in the Notebook

hmm.py is code for a Hidden Markov Model. It was used during analysis but isn't part of thr trader. It uses the hmmlearn module that needs to be installed.

