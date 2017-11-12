
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

max_ga = np.load('max_ga.npy')

avg_ga = np.load('avg_ga.npy')

max_pso = np.load('max_pso.npy')

avg_pso = np.load('avg_pso.npy')

all_generations = np.load('gen.npy')

plt.figure(1)

plt.plot(all_generations,max_pso,color = 'red',label = 'maximum fitness PSO')
plt.plot(all_generations,avg_pso,color = 'blue',label = 'average fitness PSO')
red_patch = mpatches.Patch(color = 'red',label = 'maximum fitness PSO')
blue_patch = mpatches.Patch(color = 'blue',label = 'average fitness PSO')

#plt.hold(True)
#plt.subplot(212)
plt.plot(all_generations,max_ga,color = 'green',label = 'maximum fitness GA')
plt.plot(all_generations,avg_ga,color = 'orange',label = 'average fitness GA')
plt.xlabel("Generation number")
green_patch = mpatches.Patch(color = 'green',label = 'maximum fitness GA')
orange_patch = mpatches.Patch(color = 'orange',label = 'average fitness GA')
plt.legend(handles = [red_patch,blue_patch,green_patch,orange_patch])
plt.ylabel("Fitness")


# Saving the plot
plot_name = 'plot_gen_' + sys.argv[1] + '_pop_' + sys.argv[2] + '.png'
plt.savefig(plot_name)
#plt.show()
