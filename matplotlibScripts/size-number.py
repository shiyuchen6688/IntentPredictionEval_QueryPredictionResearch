import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.patches import Ellipse, Polygon
from numpy import nan
#Set pattern array
patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')

font = {'size'   : 18}

plt.rc('font', **font)

N = 3

Types = 1
ind = np.arange(N)  # the x locations for the groups
width = 1.0/ (Types + 2)     # the width of the bars

def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .45*minsize/xsize
    ylim = .45*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

#fig, ax = plt.subplots()
fig = plt.figure()
adjustFigAspect(fig,aspect=1)
ax = fig.add_subplot(111)

maxHeight = 50

# samcube = (1000,  1000, 1000, 1000)
# rects3 = plt.bar(ind+width*0, samcube, width=width, edgecolor = '#000000', color='#FFA833', hatch="**", align="center",label='SamCube', alpha=0.75)

# samicecube = (1000,1000,1000,1000)
# rects4 = plt.bar(ind+width*1, samicecube, width=width, edgecolor = '#000000', color='#2BBAF0', hatch="o", align="center",label='NaiveTabula', alpha=0.75)




dataimport = (19,21,40)
rects1 = plt.bar(ind+width*0, dataimport, width=width, edgecolor = '#000000', color='#6666ff', hatch=".", align="center", alpha=0.75)

# routing = (291,284,284)
# rects2 = plt.bar(ind+width*1, routing, width=width, edgecolor = '#000000', color='#3366ff', hatch="+", align="center",label='Route planning', alpha=0.75)

# partitioning = (5,6,9)
# rects3 = plt.bar(ind+width*2, partitioning, width=width, edgecolor = '#000000', color='#99ff66', hatch="-", align="center",label='Vehicle partitioning', alpha=0.75)

# simulation = (520,524,887)
# rects4 = plt.bar(ind+width*3, simulation, width=width, edgecolor = '#000000', color='#00ffff', hatch="*", align="center",label='Local simulation', alpha=0.75)

# bbox_props = dict(boxstyle="circle,pad=0.1", fc="r", ec="black",alpha=1,zorder=1,lw=1)

# plt.text(0, maxHeight, 'X', ha="center", va="center", size=10, bbox=bbox_props, weight='bold')
# plt.text(1, maxHeight, 'X', ha="center", va="center", size=10, bbox=bbox_props, weight='bold')
# plt.text(2, maxHeight, 'X', ha="center", va="center", size=10, bbox=bbox_props, weight='bold')
# plt.text(3, maxHeight, 'X', ha="center", va="center", size=10, bbox=bbox_props, weight='bold')

# plt.text(0.21, maxHeight, 'X', ha="center", va="center", size=10, bbox=bbox_props, weight='bold')
# plt.text(1.21, maxHeight, 'X', ha="center", va="center", size=10, bbox=bbox_props, weight='bold')
# plt.text(2.21, maxHeight, 'X', ha="center", va="center", size=10, bbox=bbox_props, weight='bold')
# plt.text(3.21, maxHeight, 'X', ha="center", va="center", size=10, bbox=bbox_props, weight='bold')
# add some
ax.set_ylabel('Data size (GB)')
#ax.set_title('Index size on different datasets')
ax.set_xticks(ind)
ax.set_xticklabels( ('50', '100','200') )
ax.set_xlabel('Number of vehicles (thousand)')

#ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('B$^+$Tree', 'Hippo', 'BRIN', 'BRIN_32', 'BRIN_512') )
#ax.legend(loc='best', fancybox=True, framealpha=0.5)
plt.legend(bbox_to_anchor=(0,1,0, 0), loc='upper left',prop={'size':16},
           ncol=1, edgecolor='black', frameon=True, mode='', borderaxespad=0.)

ax.yaxis.grid(color='grey', linestyle='--', linewidth=0.5)

#Set display units
#label_text   = [r"$%i k$" % int(loc/10**3) for loc in plt.yticks()[0]]
#ax.set_yticklabels(label_text)

#Set Y axis range
ax.set_ylim([0,maxHeight])
plt.autoscale(axis='x')

#ax.autoscale(tight=True)
plt.savefig('size-number.pdf',bbox_inches='tight')
plt.savefig('size-number.eps',bbox_inches='tight')
plt.show(block=True)
