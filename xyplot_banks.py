#!/usr/bin/env python
import numpy as np
import argparse
import matplotlib.pyplot as plt

def plot_banks(f, title=None,banks_list=None,line_style=None):
    lines = f.readlines()

    is_first = True
    nrows = 0
    nsets = 0
    for l in lines:
        ncols = len(l.strip().split())
        if ncols == 1:
            nrows = 0
            nsets += 1
        else:
            nrows += 1

    datasets =  np.empty([nsets, nrows, 2])

    for i in range(nsets):
        for j in range(nrows):
            line = lines[i*(nrows+1) + j + 1].strip()
            x, y, err = line.split()
            datasets[i,j,:] = [x, y]

    if banks_list is None:
        banks_list = range(nsets)

    banks_list = [ int(i) for i in banks_list ]


    colors = [ 'r', 'b', 'g', 'k', 'y', 'c' ]

    for i in range(nsets):
        if i in banks_list:
            myLabel = 'bank: '+str(i)+' '+title.split('.')[0]
            #plt.plot(datasets[i,:,0], datasets[i,:,1],colors[i]+line_style,label=myLabel )
            plt.plot(datasets[i,:,0], datasets[i,:,1],line_style,label=myLabel )

    

parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='+', help="Filename with bank info")
parser.add_argument('-b', '--banks', nargs='+', help='Banks to plot')
args = parser.parse_args()
print args

banks_line_styles = { 0 : '-',
                1 : '--',
                2 : ':',
                3 : '-x',
                4 : '--x',
                5 : ':x' }

color_line_styles = { 0 : '-k',
                1 : '-r',
                2 : '-b',
                3 : '-g',
                4 : '-y',
                5 : '-c' }




for i, filename in enumerate(args.filenames):
    with open(filename, 'r') as f:
        plot_banks(f,title=filename,banks_list=args.banks, line_style=color_line_styles[i])

plt.legend()
plt.xlabel("Q (angstroms^-1")
plt.ylabel("I(Q) (arb. units)")
plt.title("Carpenter Corrections (Abs. + Mult. Scat.)")
plt.show()            
