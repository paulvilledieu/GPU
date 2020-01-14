### README ###

- How to Build the project:
  $ cd project/
  $ ./start.sh

- Run the test suite :

  For CPU (square or ball structuring elements):
  $ python3 python/run.py --cpu square --display
  $ python3 python/run.py --cpu ball --display
 
  For GPU:
  $ python3 python/run.py --gpu --display

- Run bench :

$ python3 python/bench.py

This script uses the files in bench/ dir to plot graphs.
