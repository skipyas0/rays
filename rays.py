from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageColor
import random
import numpy as np
from math import pi,inf
from time import time

""" Simple ray-tracing program
TODO: Fix bug where laser passes through a wall and 
bounces on a wall behind it, despite the first wall   
having smaller t-parameter 
"""



class Segment:
    """ Class that holds data for a segment, given by two points. Used as part of ray and obstacles. """
    def __init__(self, point_a: np.array, point_b: np.array) -> None:
        self.a = point_a
        self.b = point_b

    def draw(self, image: Image, clr: str | tuple[int,int,int] = "red") -> None:
        x_max, y_max = image.size
        xa, ya = np.clip(self.a, (0, 0), (x_max, y_max))
        xb, yb = np.clip(self.b, (0, 0), (x_max, y_max))
        d = ImageDraw.Draw(image)
        d.line((xa, ya, xb, yb), fill=clr, width=2)

    def __str__(self) -> str:
        return f"This segments goes from {self.a} to {self.b}"


class Ray:
    """ Ray object for ray-tracing """
    def __init__(self, start: np.array, vect: np.array, limit: int) -> None:
        self.start = start
        self.vect = vect
        self.limit = limit
        self.ray_segments = [Segment]

    def draw(self, image: Image) -> None:
        for segment in self.ray_segments:
            segment.draw(image)

    def create_segments(self, obstacles: list[Segment]) -> None:
        """ Given a list of obstacle segments, including image bounds, 
        calculate points of reflection and divides the ray into segments. """

        self.ray_segments = [Segment(self.start,self.start)]
        current_start = self.start
        curr_vect = self.vect  
        seg_count = 1
        next_point = self.start
        next_vect = self.start
        while seg_count < self.limit:
            least_t = inf
            # Go through all segments and find nearest mirror surface and reflection on it
            for i,segment in enumerate(obstacles):
                res = self.find_bounce_point(segment, curr_vect, current_start)
                if res != None:
                    t, inters, vect = res
                    #print(f"RUN {seg_count}, Wall {i+1} - found new point {inters} new vect {vect}, param {t}")
                    if t > 0 and t < least_t:
                        #print("pass")
                        least_t = t
                        next_point = inters   # Save point and vect for best match
                        next_vect = vect
            if least_t == inf:
                # No valid reflection found on this bounce
                break
            self.ray_segments[-1].b = next_point
            #print(f" - chose {next_point} with param {least_t}")
            curr_vect = next_vect
            current_start = next_point
            if seg_count < self.limit:
                # Prepare next segment
                self.ray_segments.append(Segment(inters, inters))
                seg_count += 1
            

    def find_bounce_point(self, obstacle_segment: Segment, current_vector, current_start):
        """ Calculates intersection of a line and a line segment, returns the line equation 
        parameter, intersection point and vector of reflected ray"""

        seg_dir = obstacle_segment.a-obstacle_segment.b
        t_mat1 = np.stack([current_start-obstacle_segment.a, seg_dir], axis = 0) 
        t_mat2 = np.stack([-current_vector, seg_dir], axis = 0) 
        t = np.linalg.det(t_mat1) / np.linalg.det(t_mat2)

        u_mat1 = np.stack([current_start-obstacle_segment.a, -current_vector], axis = 0) 
        u_mat2 = np.stack([-current_vector, seg_dir], axis = 0) 
        u = np.linalg.det(u_mat1) / np.linalg.det(u_mat2)

        if u < 0 or u > 1 or t < 1e-6:
            # u not in (0,1) means the line DOESNT intersect the segment
            # ? maybe this causes the bug TODO: try playinh with error margins in this condition
            return None
        intersection_point = check_numerical(current_start + t * current_vector)
        seg_normal = np.array([[0,1],[-1,0]]) @ seg_dir
        new_vect = check_numerical( reflect_on_bounce(current_vector, seg_normal))
        return (t, intersection_point, new_vect)

def place_laser(e):
    """ Place laser on right mouse button click. """
    global s
    image = Image.new("RGB", (w, h))
    s = np.array([e.x,e.y])
    r = Ray(s,v,lim)
    stuff = bounds+obstacles
    for seg in stuff:
        seg.draw(image, clr = "white")
    r.create_segments(stuff)
    r.draw(image)
    view = ImageTk.PhotoImage(image)
    panel.configure(image=view)
    panel.image = view

def rotate_laser(e):  
    """ Drag and rotate laser using MSB1"""
    global v
    image = Image.new("RGB", (w, h))
    v = np.array([e.x,e.y]) - s
    r = Ray(s,v,lim)
    stuff = bounds+obstacles
    for seg in stuff:
        seg.draw(image, clr = "white")
    r.create_segments(stuff)
    r.draw(image)
    view = ImageTk.PhotoImage(image)
    panel.configure(image=view)
    panel.image = view

def reflect_on_bounce(ray_vect: np.array, mirror_vect: np.array) -> np.array:
    """ Calculate reflected vector """
    norm_normal = mirror_vect / np.linalg.norm(mirror_vect)
    new_vect = ray_vect - 2*np.dot(ray_vect, norm_normal)*norm_normal
    return new_vect

def check_numerical(value: np.array, wanted_val: float = 0, margin: float = 1e-8) -> np.array:
    """ Utility to check for values near value that wouldnt otherwise count due to numerical error """
    close_indices = np.abs(value-wanted_val) < margin
    value[close_indices] = wanted_val
    return value


def create_saw(start = np.array, end = np.array, num = int, depth = int) -> list[Segment]:
    segments = []
    ln = np.linalg.norm(end-start)
    if num == 0 or all(end-start < 1e-8):
        return segments
    perpendicular_norm = np.array([[ 0, -1], [ 1, 0]]) @ ((end-start) / ln)
    step = (end-start) / num
    nx = start
    for i in range(1,num+1):
        tip = nx + step / 2 + perpendicular_norm * depth
        segments.append(Segment(nx, tip))
        nx = nx + step
        segments.append(Segment(tip, nx))
    return segments

if __name__ == "__main__" :


    root = Tk()
    w,h = 800,800
    lim = 10


    corner_a = np.array([0,0])
    corner_b = np.array([w,0])
    corner_c = np.array([w,h])
    corner_d = np.array([0,h])
    wall1 = Segment(corner_a, corner_b)
    wall2 = Segment(corner_b, corner_c)
    wall3= Segment(corner_c, corner_d)
    wall4 = Segment(corner_d, corner_a)
    bounds = [wall1, wall2, wall3, wall4]
    obstacles = create_saw(
                    np.array([100,100]), np.array([300,500]), 6, 100) + create_saw(
                            np.array([600,0]), np.array([750,500]), 6, -80)

    image = Image.new("RGB", (w, h))
    s = np.array([180,180])
    v = np.array([1,0.05])
    stuff = bounds+obstacles
    for seg in stuff:
        seg.draw(image, clr = "white")
    view = ImageTk.PhotoImage(image)
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    panel = ttk.Label(frm, image=view)
    panel.grid(column=0, row=0)

    root.bind("<B1-Motion>", rotate_laser)
    root.bind("<Button-3>", place_laser)
    root.mainloop()
