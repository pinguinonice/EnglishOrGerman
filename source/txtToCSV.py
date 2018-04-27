
input_file = open('words.txt', 'r')
output_file = open('newwords.csv', 'w')
input_file.readline()  # skip first line
for line in input_file:
    (Nummer, DE, EN) = line.strip().split('\t')
    output_file.write(','.join([Nummer, DE, EN]) + '\n')
input_file.close()
output_file.close()
