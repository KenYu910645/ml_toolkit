with open('/Users/lucky/Desktop/bdd100k/bdd100k_daytime/trainval.txt', 'w') as f:
    string = ""
    for i in range(4226):
        string += format(i, '06d') + "\n"
    f.write(string)
