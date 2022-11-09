print("Get the 64 padded binary bit of a number....\n")

def binb(x, length):
    while x >= 1:
        if x%2 != 0:
            length += 1
        x = x//2
    return length

print(binb(66992143968763904, 0))