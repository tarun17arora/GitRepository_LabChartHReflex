# importPagesLabMat

def importPagesLabMat(fileNamePath):
# This function reads matlab files exported from labchart and arrange them into channels

# calling libraries to be used
    import numpy as np
    import scipy.io

# using scipy to load matlab file. Note data is imported as arrays ndarray
    mat = scipy.io.loadmat(fileNamePath)

# extracting data
    allData = mat['data']
    allData = allData.astype(np.double)
    com = mat['com']

# extracting dataStart and dataEnd indices + defining the number of blocks
    dataStart = mat['datastart']
    dataStart = dataStart.astype(np.int32)
    dataEnd = mat['dataend']
    dataEnd = dataEnd.astype(np.int32)
    nosBlocks = dataStart.shape[1]

# note selecting good channels only (labchar labels channels without data as -1) + removing bad channels from dataStart and dataEnd + defining number of good channels
    GoodChannels = np.where(dataStart[:,0]>0)
    dataStart = dataStart[GoodChannels]
    dataEnd = dataEnd[GoodChannels]
    nosChns = len(dataStart)

# Creating a loop to generate the outcome variable allChnlData
    dataByBlock = [] #list
    new_block = []
    block = 0
    while block < nosBlocks:
        ch = 0
        while ch < nosChns:
            new_Chnl = allData[0][dataStart[ch, block]:dataEnd[ch, block]]
            new_block.append(new_Chnl)
            ch += 1
        dataByBlock.append(new_block)
        new_block = [] # note emptying the new block after adding data to dataByBlock to prepare for the new block
        block += 1
    return(dataByBlock, com)

################################################################################################
## starting another function here - getPagesInfo ##
################################################################################################

def get_page_info(com, samp_rate, pre_time_ms, post_time_ms):
    
    #getPageInfo will create a column with page number information 
    #Note: this will be done separately for each block
    
    #importing numpy library
    import numpy as np    
    
    # creating start and end points for pages
    # converting time_ms to data_points
    
    pre_time_datapoint = samp_rate * pre_time_ms / 1000
    post_time_datapoint = samp_rate * post_time_ms / 1000
    
    mod_com = com[:, [1, 2]]
    mod_com = np.column_stack((mod_com, mod_com[:, 1] - pre_time_datapoint))  # adding page start information to modified com
    mod_com = np.column_stack((mod_com, mod_com[:, 1] + post_time_datapoint))  # adding page end information to modified com
    
    # creating pagesByBlock to store information in cells for each block
    
    # setting up an empty list to store information
    unique_blocks = np.unique(mod_com[:, 0])
    pages_by_block = []
    
    # running a loop to add information to the empty list
    for pg_block_no in unique_blocks:
        ind_block_ind = np.where(mod_com[:, 0] == pg_block_no)[0]  # finding indices of required block
        ind_block_vals = mod_com[ind_block_ind,2:4]
        pages_by_block.append(ind_block_vals)
    return pages_by_block


################################################################################################
## starting another function here - markPages ##
################################################################################################

def mark_pages(filename,samp_rate, pre_time_ms, post_time_ms):
# markPages labels the entire data with page information
# note: this function will add another column to the imported data with the channel
# information "page label"

    #importing numpy library
    import numpy as np 

    # calling the function "read_lab_chart_matfile" to obtain the data sorted by blocks ("data_by_block") and 
    # the information regarding trigger times ("com") that will be used to create info about page numbers

    (data_by_block, com) = importPagesLabMat(filename)
    nos_blocks = len(data_by_block)

    # calling the function "pages_by_block" to obtain information about page numbers
    pages_by_block = get_page_info(com, samp_rate, pre_time_ms, post_time_ms)

    # creating marked_pages as a copy of data_by_block for further modifications
    marked_pages = data_by_block.copy()
    unique_blocks = np.unique(com[:, 1])
    block_no = 0
    pg_no = 0  # this is page within a block
    given_page_no = 1  # this is labChart page number - note this so that page number doesn't go back to 1 for each new block

    while block_no < nos_blocks:

    # extracting data and page number information for the block
        pages_info_for_block = pages_by_block[block_no]
        data_for_block = data_by_block[block_no]

        # determining the "number of the pages" and "data point in each page" within the current block
        pages_in_block = pages_info_for_block.shape[0] #determining the number of pages within this block
        block_data_length = len(data_for_block[block_no])  # determing the number of datapoints in each channel for this block 

        # creating a zero matrix of size of block length
        pagelabel_shape = np.shape(data_for_block[0]) #note: specifying the shape of the ndarray to be created 
        page_label = np.zeros(pagelabel_shape)
        
        if pages_in_block <= 0:
            print("invalid number of pages in this block")
        
        elif pages_in_block > 0:  
            while pg_no < pages_in_block:
                st = int(pages_info_for_block[pg_no, 0])
                end = int(pages_info_for_block[pg_no, 1] + 1)
                page_label[st:end] = given_page_no
                pg_no += 1
                given_page_no += 1
            page_label = page_label.tolist() #converting ndarray to list for compatibility
            data_for_block.append(page_label)
            marked_pages[block_no] = data_for_block
            pg_no = 0  # resetting page number to be read to 1
            block_no += 1

    return marked_pages

################################################################################################
## starting another function here - getPagesData ##
################################################################################################


def get_pages_data(filename, samp_rate, pre_time_ms, post_time_ms, avgFact = 0):
    # getPagesData will obtain LabChart pages in a cell #
    # note: the output will have an extra channel with the page number values only

    #importing numpy library
    import numpy as np 
    
    # Getting data sorted by blocks 
    marked_pages = mark_pages(filename,samp_rate, pre_time_ms, post_time_ms)
    
    club_data = np.hstack(marked_pages)
     
    # Identify page numbers and remove zeros
    page_nos = np.unique(club_data[-1])
    page_nos = page_nos[page_nos > 0] # note: selecting data that belongs to pages
    
    #starting a loop to create pages with all channels information in that page
    pages = []
    for pg_no in page_nos:
        ind_pg = np.where(club_data[-1] == pg_no)[0] #finding the indices of data belonging to current page 
        pg_val = club_data[:,ind_pg] #double check if there is a need to switch columns and rows
        pages.append(pg_val) #maybe first convert to list before appending?
    
    # OPTIONAL: averaging the pages if asked by the user
    if avgFact != 0:
        a1 = 0
        diff = avgFact
        noLots = int(len(pages) / avgFact) # this gives number of resulting pages after averaging
        chNos = len(pages[0]) # this gives number of channels in the first page - should be common to all pages
        new_lotArray = []
        for lotNo in range(1,noLots+1): #double check +1 correction
            st_ind = a1 + (lotNo - 1)*diff #using arithmetic mean
            end_ind = st_ind + diff 
            useData = pages[st_ind:end_ind]
            mean_ChArray = []
            for chNo in range(0, chNos):
                pg_array4mean = []
                for pgNo in range(0, len(useData)):
                    ind_Chnl = useData[pgNo][chNo]
                    pg_array4mean.append(ind_Chnl)
                mean_pgArray = np.mean(pg_array4mean,axis = 0)
                mean_ChArray.append(mean_pgArray)
            new_lotArray.append(mean_ChArray)
        pages = new_lotArray

    return pages

################################################################################################
## starting another function here - plotAndExtract ##
################################################################################################
def plotAndExtract(filename, sampRate, preTime, postTime, avgFact = 0):
    import numpy as np
    from Function_readLabChartMat import importPagesLabMat, get_page_info, mark_pages, get_pages_data
    if avgFact == 1:
        print('Note: Averaging of Pages NOT PERFORMED')
    else:
        print('Note: Each Page is an average of following number of pages = ' + str(avgFact))

# calling get_pages_data to obtain the pages
    pages = get_pages_data(filename, sampRate, preTime, postTime, avgFact)

    no_pages = len(pages)
    print('Note: Total number of Pages from LabChart = ' + str(no_pages))

    no_chnls = len(pages[0])
    print('Note: Total number of Channels from LabChart = ' + str(no_chnls-1))

# Plotting Functions
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    timeax = np.linspace(-preTime, postTime, len(pages[0][0]))

    ch = 1
    for pg in range(no_pages):
        plt.plot(timeax, pages[pg-1][ch-1], color = 'blue')
        plt.autoscale()

    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title('All Pages Over Time')
    plt.grid(False)

# Obtain user input
    user_input = plt.ginput(n=4, timeout=-1)
# mTimeRange = user_input[0:1]
# hTimeRange = user_input[2:3]
# Print the user input
# print("M-wave Range:", mTimeRange)
# print("H-wave Range:", hTimeRange)

    # drawing vertical lines at user input
    # for point in user_input:
    #     x_coord = point[0]  # Extract x-coordinate
    #     plt.vlines(x=x_coord, ymin=min(timeax), ymax=max(timeax), color='red', linestyle='--')
# Show the plot
    plt.show()
# Performing Analysis for the Calculation of Outcome Variables
    xvals = []  
    for xind in range(len(user_input)):
        xvals.append(user_input[xind][0])
    
    mtimerange = xvals[:2]
    htimerange = xvals[-2:]

    return mtimerange,htimerange

################################################################################################
## starting another function here - backRMS ##
################################################################################################
def backRMS(indEpoch, RMS_Signal_StartTime_ms, RMS_Signal_EndTime_ms, RMS_samp_rate, RMS_pre_time_ms):
    
    # assuming the user wants to calculate background rms of the signal ranging from...
    # RMS_Signal_StartTime_ms (e.g., 55ms) to the RMS_Signal_EndTime_ms (5 ms) where 0 is the electrical stimulus 
    # NOTE: no need to add a negative sign to the start or end time as it automatically assumes it occurs before 0ms mark

    import numpy as np 
    
    RMS_startTime = pre_time_ms - RMS_Signal_StartTime_ms
    RMS_startPoint = int(RMS_startTime * samp_rate/1000)
    
    RMS_endTime = pre_time_ms - RMS_Signal_EndTime_ms
    RMS_endPoint = int(RMS_endTime * samp_rate/1000)
    
    RMS_signal = indEpoch[RMS_startPoint:RMS_endPoint]
       
    return(RMS_signal)

################################################################################################
## starting another function here - averaging values by an avgFact
################################################################################################
# this can be used to average the background rms values by certain fact

def avgVals(val_list, avgFactor):

    import numpy as np
    
    a1 = 0
    diff = avgFactor
    noLots = int(len(val_list) / avgFactor) # this gives number of resulting pages after averaging
    
    avgRmsVals = []

    for lotNo in range(1,noLots+1): #double check +1 correction

        st_ind = a1 + (lotNo - 1)*diff #using arithmetic mean
        end_ind = st_ind + diff 

        useData = val_list[st_ind:end_ind]
        mean_ChArray = np.mean(useData)
        avgRmsVals.append(mean_ChArray)

    return avgRmsVals

################################################################################################
## starting another function here - automatic event detection 
################################################################################################

# this function automatically detects the onset and offset of H and M waves

def autoEventDetection(indEpoch, backgroundEMG, samp_rate, pre_time_ms = 50, onTime_ms = 5, offTime_ms = 5, std = 3):
    # ind Epoch is the signal to be tested
    # background EMG is the reference EMG for the baseline value
    # samp_rate is the acquisition sampling rate
    # pre_time is the time point from the starting of the signal til the trigger (electrical stimulation)
    # std - standard deviation to calculate threshold value defined as mean(background EMG) + std* standard deviation(backgroundEMG)
    # onTime_ms - time period for which the test signal should go above the threshold value to be detectes as event ON
    # offTime_ms - time period for which the test signal should remain below the threshold value to be detected as event OFF
    import numpy as np

    expectedMOnset_ms = 5 # expected on and off are set default to 5 and 35 for the soleus muscle
    expectedHOnset_ms = 35

# removing DC offset
    indEpoch_noDCOffset = indEpoch - np.mean(backgroundEMG)

    # signal rectification
    rect_signal = np.abs(indEpoch_noDCOffset)

    # onset detection starts here

    threshold_signal = np.abs(backgroundEMG - np.mean(backgroundEMG))
    threshold_amp = np.mean(threshold_signal) + np.std(threshold_signal)*std   

    onTime_points = onTime_ms * samp_rate/1000 
    offTime_points = offTime_ms * samp_rate/1000

    boolean_Array = rect_signal > threshold_amp
    switchON_counter = 0
    switchOFF_counter = 0
    on_indices = []
    off_indices = []
    eventActive = False  

    for i in range(len(boolean_Array)):
        if boolean_Array[i]:
            switchON_counter += 1
            switchOFF_counter = 0
            if switchON_counter == onTime_points:
                on_index = i - onTime_points + 1
                on_indices.append(on_index)
                eventActive = True
                switchON_counter = 0
        else:
            switchON_counter = 0  
            switchOFF_counter += 1

    # Detect event off only if an event on has already occurred
            if switchOFF_counter == offTime_points and eventActive:
                off_index = i - offTime_points + 1
                off_indices.append(off_index)  # Event off detected
                eventActive = False  # Mark event as inactive
                switchOFF_counter = 0  # Reset event off counter
    
    # if more than 2 onset detected
    if on_indices == []:
        print("no H or M events detected")

    elif len(on_indices) == 1:
        print("only single event detected")

    elif len(on_indices) == 2:
        print("2 onset found, probably one each for H and M!")

    elif len(on_indices) > 2:        
        print("more than 2 onset found, reporting closest to the expected values of 5ms and 35 ms")

        expected_vals = (np.array([expectedMOnset_ms, expectedHOnset_ms]) + pre_time_ms)* samp_rate/1000
        
        detected_Mval = min(on_indices, key=lambda x: abs(x - expected_vals[0]))
        detected_Hval = min(on_indices, key=lambda x: abs(x - expected_vals[1]))

        on_indices = np.array([detected_Mval, detected_Hval])
    
    return(on_indices, off_indices)

    
################################################################################################
## starting another function here - manual event detection
################################################################################################

def manualEventDetect(indEpoch, pre_time_ms, samp_rate, hOnset_ms, mOnset_ms, rangeTime_ms):
    
    # indEpoch: emg epoch of the signal to be evaluated
    # hOnset_ms: expected onset of H-wave in ms
    # mOnset_ms: expected onset of M-wave in ms
    # rangeTime_ms: rough estimate of how many ms after the onset the signal will last without any other event
    # samp_rate is the acquisition sampling rate
    # pre_time is the time point from the starting of the signal til the trigger (electrical stimulation)

    import numpy as np

    hRange_ms = np.array([hOnset_ms, hOnset_ms+rangeTime_ms]) + pre_time_ms
    mRange_ms = np.array([mOnset_ms, mOnset_ms+rangeTime_ms]) + pre_time_ms

    hRange_points = (hRange_ms*samp_rate//1000)
    mRange_points = (mRange_ms*samp_rate//1000)

    hSignal = np.array(indEpoch[hRange_points[0]:hRange_points[1]])
    mSignal = np.array(indEpoch[mRange_points[0]:mRange_points[1]])

    return(hSignal, mSignal)
    