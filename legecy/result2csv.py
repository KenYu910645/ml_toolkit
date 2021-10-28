import os

input_path = os.path.join("/Users/lucky/Desktop/bdd100k", 'training_proces.txt')
output_path = os.path.join("/Users/lucky/Desktop/bdd100k", 'training_proces.csv')
a = []

with open(input_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        total_iter = line.split()[7]
        loss = line.split()[10]
        a.append((total_iter, loss))
        #print(total_iter)
        # print(loss)

with open(output_path, 'w') as f:
    for i in a :
        f.write(i[0] + ',' + i[1] + '\n')


