file_dir = './txt_lists/'
file_name= 'sketch_test.txt'

with open(file_dir + file_name, 'r') as file:
    text = file.readlines()

j = 1
for i in range(len(text)):
    text.insert( j, text[j-1][:-7] + '_1' + text[j-1][-7:])
    j += 2

with open('new_'+file_name, 'w') as file:
    file.writelines(text)
