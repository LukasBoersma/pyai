reset
list = system('ls data/momentum*')

set term png
set output 'momentum.png'

set datafile separator ";"


plot for [file in list ] file using 1:2 with lines title sprintf(file)
