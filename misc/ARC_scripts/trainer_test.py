import sys
import numpy

def main(args):
    
    split = args.split()
    lr = float(split[0])
    dropout = float(split[1])
    
    text_file = open("./sample.txt", "w")
    n = text_file.write(str(lr))
    text_file.close()
    return None

if __name__ == '__main__':
    args = sys.stdin.read()
    main(args)