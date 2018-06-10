from Tkinter import *

class TextEditor:
    def __init__(self):
        self.ROOT=Tk()
        self.ROOT.title("Hand Gesture Text Editor")
        self.CONTAINER=Text(self.ROOT)
        self.CONTAINER.insert(END, "")
        self.CONTAINER.grid()
        self.ROOT.update()

    def insert(self, letter):
        self.CONTAINER.insert(END, letter)
        self.ROOT.update()

    def delete(self):
        self.CONTAINER.delete("end-2c")
        self.ROOT.update()

