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