from memetic.memetic import memetic
from memetic.utils import *
import sys

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print("Path missing")
        sys.exit()

    path = args[1]
    
    cost, W, H = memetic(path, generate=True)
    print(cost)

