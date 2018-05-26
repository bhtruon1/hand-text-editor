import csv
import numpy as np
import os

# dataset directory: './dataset/SL/Dataset'


class ImageDataLoader:
    def __init__(self):
        self.dir = "./dataset/SL/Dataset/"
        self.file_type = ".JPG"
        self.categories = {}
        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.out_array = []

    def initiate_categories(self):
        for i in self.classes:
            self.categories[i] = []

    def load_single_category(self, i):
        """
        i refers to the specific subdirectory we are loading
        ex. i = "4" or i = "8"
        """
        for file in os.listdir(self.dir + i):
            if file.endswith(self.file_type):
                self.categories[i].append(os.path.join(self.dir + i, file))

    def load_categories(self):
        for i in self.classes:
            self.load_single_category(i)

    def create_out_array(self):
        for key, value in self.categories.items():
            for i in value:
                self.out_array.append([i, key])

    def write_to_csv(self):
        with open('example.csv', 'w') as File:
            wr = csv.writer(File, dialect="excel")
            wr.writerows(self.out_array)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

