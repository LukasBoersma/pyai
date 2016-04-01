#import sys
#add parent folder to path
# http://stackoverflow.com/a/16780068
#sys.path.append("../")
import subprocess
#from subprocess import call

import re

#import argparse
#parser = argparse.ArgumentParser("Tests parameters")

epochs = [1,3,6,10,12,15,18,20,25,30,40,50,60,100,150,175]
#epochs = [1,2]
def test(parameter_name, values, filename = "../data500.gz"):
    total_performance = []
    for v in values:
        current_performance=[]
        for epoch in epochs:
            print('Instance' + str(v) + ' ' + str(epoch))
            out = subprocess.check_output(['ipython',         # name of programm
                '../learn_and_play.py',     # script to call
                '--',                       # all arguments beyond that (that are given with '--') are handed to script, not ipython
                '--epochs',
                str(epoch),
                '--training_file',
                filename,
                '--'+ parameter_name,           # dashes of next parameter +  parameter name
                str(v) ])       #parameter value

            regex = re.compile('Wins: ([0-9]*)')
            wins = regex.findall(out)[0]
            current_performance.append([v, epoch, wins])
        total_performance.append(current_performance)

    for performance in total_performance:
        f = open('data/' + parameter_name + '_' + str(performance[0][0]) + '.txt','w')
        for triple in performance:
            f.write( str(triple[1]) + ';' + str(triple[2]) + '\n')
        f.close()

#test('momentum' ,[0.1,0.2])
test('momentum' ,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
