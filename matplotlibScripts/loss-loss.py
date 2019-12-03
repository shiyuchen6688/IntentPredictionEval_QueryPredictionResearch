import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.patches import Ellipse, Polygon
from numpy import nan
#Set pattern array
patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')

font = {'size'   : 18}

plt.rc('font', **font)

N = 4

Types = 5
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
adjustFigAspect(fig,aspect=1.5)
ax = fig.add_subplot(111)


samfirst1 = (353.37,353.37,353.37,353.37)
# rects1 = plt.bar(ind, samfirst1, width=width, edgecolor = '#000000', color='#6666ff', hatch="", align="center",label=r'SamFirst-10$^5$', alpha=0.75)
rects1 = plt.errorbar(ind, samfirst1, yerr=[[353.31,353.31,353.31,353.31], [146.63,146.63,146.63,146.63]], fmt='', linestyle='',label=r'SamFirst-10$^5$', color='#6666ff', capthick= 3,capsize=3, markersize=10)
rects1[-1][0].set_linestyle('-')

samfirst2 = (325.00,325.00,325.00,325.00)
# rects2 = plt.bar(ind+width, samfirst2, width=width, edgecolor = '#000000', color='#3366ff', hatch="", align="center",label=r'SamFirst-10$^6$', alpha=0.75)
rects2 = plt.errorbar(ind+width, samfirst2, yerr=[[324.99,324.99,324.99,324.99], [175.00,175.00,175.00,175.00]], fmt='', linestyle='', label=r'SamFirst-10$^6$',color='#3366ff', capthick= 3,capsize=3, markersize=10)
rects2[-1][0].set_linestyle('-')

samlater = (1.2,0.72,0.45,0.24)
# rects3 = plt.bar(ind+width*2, samlater, width=width, edgecolor = '#000000', color='#99ff66', hatch="", align="center",label='SamFly', alpha=0.75)
rects3 = plt.errorbar(ind+width*2, samlater, yerr=[samlater, [0.77,0.28,0.05,0.03]], fmt='^', linestyle=':', label='SamFly', color='k', markeredgecolor = 'k', markerfacecolor = '#99ff66', capthick= 3, capsize=3, markersize=8)
rects3[-1][0].set_linestyle('-')

poisam = (1.29,0.78,0.45,0.24)
# rects4 = plt.bar(ind+width*3, poisam, width=width, edgecolor = '#000000', color='#cccc00', hatch="", align="center",label='POIsam', alpha=0.75)
rects4 = plt.errorbar(ind+width*3, poisam, yerr=[[1.23,0.76,0.44,0.23], [4.63,2.23,0.88,0.59]], fmt='s', linestyle=':', label='POIsam', color='k', markeredgecolor = 'k', markerfacecolor = '#cccc00',capthick= 3, capsize=3, markersize=10)
rects4[-1][0].set_linestyle('-')

samicecubeopt = (0.45,0.26,0.16,0.10)
# rects5 = plt.bar(ind+width*4, samicecubeopt, width=width, edgecolor = '#000000', color='#FF0101', hatch="", align="center",label='Tabula', alpha=0.75)
rects5 = plt.errorbar(ind+width*4, samicecubeopt, yerr=[samicecubeopt, [1.54,0.72,0.34,0.15]], fmt='*', linestyle=':', label='Tabula', color='k', markeredgecolor = 'k', markerfacecolor = '#FF0101', capthick= 3, capsize=3, markersize=12)
rects5[-1][0].set_linestyle('-')

maxHeight = 6

bbox_props = dict(boxstyle="sawtooth,pad=0.3", fc="pink", ec="black",alpha=1,zorder=1,lw=1)

plt.text(0-0.02, maxHeight, 'Null', ha="center", va="center", size=10, bbox=bbox_props, weight='bold', rotation=-90)#70%
plt.text(1-0.02, maxHeight, 'Null', ha="center", va="center", size=10, bbox=bbox_props, weight='bold', rotation=-90)
plt.text(2-0.02, maxHeight, 'Null', ha="center", va="center", size=10, bbox=bbox_props, weight='bold', rotation=-90)
plt.text(3-0.02, maxHeight, 'Null', ha="center", va="center", size=10, bbox=bbox_props, weight='bold', rotation=-90)

plt.text(0.2, maxHeight, 'Null', ha="center", va="center", size=10, bbox=bbox_props, weight='bold', rotation=-90)#65%
plt.text(1.2, maxHeight, 'Null', ha="center", va="center", size=10, bbox=bbox_props, weight='bold', rotation=-90)
plt.text(2.2, maxHeight, 'Null', ha="center", va="center", size=10, bbox=bbox_props, weight='bold', rotation=-90)
plt.text(3.2, maxHeight, 'Null', ha="center", va="center", size=10, bbox=bbox_props, weight='bold', rotation=-90)

# add some
ax.set_ylabel('Actual accuracy loss (dollar)')
#ax.set_title('Index size on different datasets')
ax.set_xticks(ind+2*width)
ax.set_xticklabels( ('2.0', '1.0','0.5', '0.25') )
ax.set_xlabel('Accuracy loss threshold (dollar)')

#ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('B$^+$Tree', 'Hippo', 'BRIN', 'BRIN_32', 'BRIN_512') )
#ax.legend(loc='best', fancybox=True, framealpha=0.5)
plt.legend(bbox_to_anchor=(0.,0.9, 1, 0), loc='upper right',prop={'size':16},
           ncol=2, edgecolor='black', frameon=False, mode='', borderaxespad=0.)

ax.yaxis.grid(color='grey', linestyle='--', linewidth=0.5)

#Set display units
#label_text   = [r"$%i k$" % int(loc/10**3) for loc in plt.yticks()[0]]
#ax.set_yticklabels(label_text)

#Set Y axis range
ax.set_ylim([0,maxHeight])
plt.autoscale(axis='x')

# ax.set_yscale('log')
# ax.set_yscale('symlog', linthreshx = 0.001)

#ax.autoscale(tight=True)
plt.savefig('loss-loss.pdf',bbox_inches='tight')
plt.savefig('loss-loss.eps',bbox_inches='tight')
plt.show(block=False)
