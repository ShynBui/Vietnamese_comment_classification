# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pyvi import ViTokenizer, ViPosTagger
import py_vncorenlp
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(os.getcwd())
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir= os.path.join(os.getcwd(), 'processe'))
    text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
    output = rdrsegmenter.word_segment(text)
    print(output)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
