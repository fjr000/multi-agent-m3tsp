from collections import deque

import numpy as np

class CourseController(object):
    def __init__(self):
        self.q = deque(maxlen=10)
        self.course_list = [
            (1, 5), (2, 10), (3, 15), (4, 20), (5, 25), (6, 30), (7, 35), (8, 40), (9, 45), (10, 50),
            (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90), (10, 100),
            (1, 20), (2, 40), (3, 60), (4, 80), (5, 100), (6, 120), (7, 140), (8, 160), (9, 180), (10, 200),
            (1, 30), (2, 60), (3, 90), (4, 120), (5, 150), (6, 180), (7, 210), (8, 240), (9, 270), (10, 300),
            (1, 40), (2, 80), (3, 120), (4, 160), (5, 200), (6, 240), (7, 280), (8, 320), (9, 360), (10, 400),
            (1, 50), (2, 100), (3, 150), (4, 200), (5, 250), (6, 300), (7, 350), (8, 400), (9, 450), (10, 500),
        ]
        self.course = 0

    def get_course(self):
        return self.course_list[self.course]

    def update_result(self, gap):
        self.q.append(gap)
        if_next = np.count_nonzero(np.array(list(self.q)) < 0.1) >= 8
        if if_next:
            self.q.clear()
            self.course = (self.course+1)%len(self.course_list)