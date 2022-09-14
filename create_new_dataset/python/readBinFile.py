import struct


def read_bin_float(filename):
    with open(filename, mode='rb') as file: # b is important -> binary
        fileContent = list(file.read())
    n= int(len(fileContent)/4)
    with open(filename, mode='rb') as file:
        return struct.unpack('f' * n, file.read(4 * n))


def read_bin_16bit(filepath):
    with open(filepath, mode='rb') as file: # b is important -> binary
        fileContent = list(file.read())
    n=int(len(fileContent)/2)
    with open(filepath, mode='rb') as file:
        data=struct.unpack('H'*n, file.read(2*n))
    return data


def read_bin_double(filepath):
    with open(filepath, mode='rb') as file: # b is important -> binary
        fileContent = list(file.read())
    n=int(len(fileContent)/8)
    del fileContent
    with open(filepath, mode='rb') as file:
        data=struct.unpack('d'*n, file.read(8*n))
    return data
