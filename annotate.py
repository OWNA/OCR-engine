import cv2
import csv
import os

counters = xrange(0, 114)

CELL_NUMBER = 1
CELL_DESCRIPTION = 2
CELL_OTHER = 3
CELL_POSSIBLE_NUMBER = 4


class Cell:
    def __init__(self, l, r, t, b, content, cell_type):
        self.l = l
        self.r = r
        self.t = t
        self.b = b
        self.filename = ''
        self.content = content
        self.cell_type = cell_type

    def is_number(self):
        return self.cell_type == CELL_NUMBER

    def is_possible_number(self):
        return self.cell_type == CELL_POSSIBLE_NUMBER

    def is_description(self):
        return self.cell_type == CELL_DESCRIPTION

    def is_other(self):
        return self.cell_type == CELL_OTHER


def draw_image_cells(image, filename, cells):
    image_f = image.copy()
    image_c = image.copy()
    num = 0
    for cell in cells:
        xL = cell.l
        xR = cell.r
        yT = cell.t
        yB = cell.b
        x1, y1 = int(xL), int(yT)
        x3, y3 = int(xR), int(yB)
        if cell.is_number():
            color = (0, 255, 0)
        elif cell.is_possible_number():
            color = (255, 0, 255)
        elif cell.is_description():
            color = (0, 255, 255)
        elif cell.is_other():
            color = (0, 0, 255)
        cv2.rectangle(image_f, (x1, y1), (x3, y3), color, -1)
        num += 1
    image_f = image_c * 0.4 + image_f * 0.6
    image_f = cv2.resize(image_f, (0, 0), fx=0.3, fy=0.3)
    cv2.imwrite(filename, image_f)


def draw_cells(image_filename, counter, cells):
    image = cv2.imread(image_filename, 3 | cv2.IMREAD_IGNORE_ORIENTATION)
    out_filename = 'annotated/photo-{}.png'.format(counter)
    draw_image_cells(image, out_filename, cells)


image_filenames = [
    'images/photo-{counter}.jpg'.format(counter=counter)
    for counter in counters
]


def read_cells(file_name):
    cells_by_filename = {}

    with open(file_name) as cells_file:
        reader = csv.DictReader(cells_file)
        for row in reader:
            filename = row['filename']
            if filename not in cells_by_filename:
                cells_by_filename[filename] = []
            cell = Cell(
                int(row['l']),
                int(row['r']),
                int(row['t']),
                int(row['b']),
                row['content'],
                int(row['cell_type']),
            )
            cells_by_filename[filename].append(cell)
    return cells_by_filename


if __name__ == '__main__':
    cells_by_filename = read_cells('elements.dat')
    for image_filename, counter, in zip(image_filenames, counters):
        if os.path.isfile(image_filename):
            draw_cells(image_filename, counter,
                       cells_by_filename[image_filename])
