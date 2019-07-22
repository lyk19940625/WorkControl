
import cv2
import csv
def readline():
    lines = []

    with open('line.csv','r') as file:
        csv_data = csv.reader(file)
        for row in csv_data:
            line = []
            for r in row:
                line.append(int(r))
            lines.append(line)
    return lines

def draw(frame):
    line = readline()
    for i in range(len(line)-1):

        p1 = (int(line[i][0]),int(line[i][1]))
        p2 = (int(line[i+1][0]),int(line[i+1][1]))
        cv2.line(frame, p1, p2, (0, 255, 255), 3)