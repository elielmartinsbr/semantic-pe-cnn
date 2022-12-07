# Separate PE files into semantic parts

def read_file_gz(file_path):
    import gzip
    """Read the binary sequence of a file."""
    f = gzip.open(file_path, 'r')
    return f.read()

def read_file(file_path):
    with open(file_path, "rb") as binary_file:
        return binary_file.read()

def get_all_data_from_pe(filename):
    import lief
    import numpy as np
    import pathlib

    if '.gz' in pathlib.Path(filename).suffixes:
        byte_stream = read_file_gz(filename)
    else:
        byte_stream = read_file(filename)

    # Converts from hex to decimal
    dataList=list(byte_stream)

    try:
        binary_lief=lief.PE.parse(raw=dataList) # class Binary
    except:
        print('error in %s' % filename)
        return np.array([0]),np.array([0]),np.array([0]),np.array([0])

    header=[]
    code=[]
    data=[]
    try:
        header=dataList[0:binary_lief.sizeof_headers]
    except:
        header=dataList[0:4096]

    try:
        for section in binary_lief.sections: #class Section

            if (lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE or
                lief.PE.SECTION_CHARACTERISTICS.CNT_CODE)  in section.characteristics_lists:
                code.extend(section.content)
                continue
            if (lief.PE.SECTION_CHARACTERISTICS.CNT_INITIALIZED_DATA or
                lief.PE.SECTION_CHARACTERISTICS.CNT_UNINITIALIZED_DATA)  in section.characteristics_lists:
                data.extend(section.content)
    except:
        print('error in %s' % filename)


    header_array=np.trim_zeros(np.array(header, dtype=np.uint8))
    code_array=np.trim_zeros(np.array(code, dtype=np.uint8))
    data_array=np.trim_zeros(np.array(data, dtype=np.uint8))
    file_array=np.trim_zeros(np.array(dataList, dtype=np.uint8))

    return header_array,code_array,data_array,file_array
