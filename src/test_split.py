import os
import random
import shutil
training_set = os.listdir('edgedata/training2/')

for letter in training_set:
    file_list = os.listdir('edgedata/training2/' + str(letter))
    data_size = len(file_list)
    print(letter)
    test_list = random.sample(file_list, k=200)
    test_files = ['edgedata/training2/' + letter + '/' + x for x in test_list]
    move_to = "edgedata/testing2/" + letter + '/'
    if not os.path.exists(move_to):
        os.makedirs(move_to)
    for i in test_files:
        try:
            shutil.move(i, move_to)
        except:
            print(move_to)

