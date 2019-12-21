import sys
import os
import csv



if __name__ == "__main__":
    args = sys.argv
    csv_file = args[1]
    with open(os.path.join(os.getcwd(), csv_file), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AP50', 'AP75', 'mAP'])
