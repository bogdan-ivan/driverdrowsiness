

def data_harvester(arrayOfData,fileName):
    """This function gets and array of data and writes it in a .csv file for future procesing"""
    for data in arrayOfData:
        fileName.write(str(data)+",")
    fileName.write("\n")