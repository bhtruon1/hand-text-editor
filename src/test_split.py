import os
import random
import shutil
# training_set = os.listdir('training/Letters')
#
# for letter in training_set[1:]:
#     if letter != 'J':
#         continue
#     file_list = os.listdir('training/Letters/' + str(letter))
#     data_size = len(file_list)
#     test_list = random.sample(file_list, k=200)
#
#     test_files = ['training/Letters/' + letter + '/' + x for x in test_list]
#     # print(test_files)
#     move_to = "testing/" + letter + '/'
#     for i in test_files:
#         try:
#             shutil.move(i, move_to)
#         except:
#             pass
dir = '/Users/chenmo/Files/PythonProjects/cs175_project/hand-text-editor/src/edgedata/training/'
training_set = os.listdir(dir)

print(training_set)
for letter in training_set[1:]:

    file_list = os.listdir(dir + str(letter))
    data_size = len(file_list)
    test_list = random.sample(file_list, k=int(data_size/2))

    to_remove = [dir + letter + '/' + x for x in test_list]

    for f in to_remove:
        os.remove(f)
