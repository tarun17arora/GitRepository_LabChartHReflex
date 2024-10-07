import numpy as np
from Function_readLabChartMat import importPagesLabMat, get_page_info, mark_pages, get_pages_data
filename = '/Users/tarunarora/Documents/MATLAB/EPD01/MAT Files/EPD01_HMCurve_5trials_22042024.mat'

# input values for variables to be used later
sampRate = 2000
preTime = 50
postTime = 100
avgFact = 5 #this is for averaging the pages to obtain a single page

# calling get_pages_data to obtain the pages
pages = get_pages_data(filename, sampRate, preTime, postTime, avgFact)

no_pages = len(pages)
print('Note: Total number of Pages from LabChart = ' + str(no_pages))

no_chnls = len(pages[0])
print('Note: Total number of Channels from LabChart = ' + str(no_chnls-1))



# Plotting Pages
import matplotlib as mpl
import matplotlib.pyplot as plt
timeax = np.linspace(-preTime, postTime, len(pages[0][0]))


# getting inputs for plotting
# pg = 1
# ch = 1
# y_line = pages[pg-1][ch-1]

# fig, ax = plt.subplots (1,1)
# plt.plot(timeax, pages[pg-1][ch-1], color = 'orange')
# plt.ylim(-0.1,0.1)
# plt.show()


ch = 1
for pg in (1,no_pages):
    plt.plot(timeax, pages[pg-1][ch-1], color = 'orange')
    plt.autoscale()
plt.show()
# for pg in 2,11:
#     ax.plot(timeax, pages[pg-1][ch-1], color = 'black')
# plt.show()

# def on_click (event):
#     print(event)
# fig.canvas.mpl_connect(
#     'button_press_event', on_click)


# # trying interactive plot

# from matplotlib.widgets import Slider

# fig, ax = plt.subplots(2,1)
# ax.autoscale()
# plt.subplots_adjust(bottom = 0.25)
# y_val = pages[pg-1][ch-1]

# l, = ax.plot(timeax, y_val, lw = 2)
# ax_slider = plt.axes(arg=[0.25, 0.1, 0.65, 0.03])

# slider = Slider(ax_slider,label ='Page', valmin = 1, valmax = no_pages, valinit = pg)

# def update(vallue):
#     pg = slider.val
#     l.set_ydata(timeax, y_val, lw = 2)
#     fig.canvas.draw_idle()

# slider.on_changed(update)

# plt.show
