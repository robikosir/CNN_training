import csv
import matplotlib.pyplot as plt

gender_acc = []
race_acc = []
with open('AWE.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            gender_acc.append(float(row[1])*100)
            race_acc.append(float(row[3])*100)
        line_count += 1

print(max(gender_acc))
print(max(race_acc))
line1, = plt.plot(gender_acc, label='Gender')
line2, = plt.plot(race_acc, label='Race')
plt.legend(handles=[line1, line2])
plt.ylabel('Accuracy [%]')
plt.xlabel('Epoch')
plt.show()


