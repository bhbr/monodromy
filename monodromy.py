#!/usr/bin/env python2.7

import pygame as pg
import math
import numpy as np
import numpy.polynomial.polynomial as npp
import numdifftools as nd
from matplotlib.colors import hsv_to_rgb
from scipy.optimize import root

TAU = 2*np.pi
dim = 451
scale = 1
dim_scaled = int(scale * dim)
circle_size = 5
font_size = 30

values = np.ndarray((dim,dim))
HSV = RGB = np.ndarray((dim,dim,3))
HSV[:,:,1] = 1
#RGB_scaled = np.ndarray((2*dim,2*dim,3))
mySurf = pg.Surface((dim,dim))
mySurf_scaled = pg.Surface((dim_scaled,dim_scaled))

pg.init()


xmin = ymin = -2
xmax = ymax = 2




(width, height) = (dim_scaled,dim_scaled)
window = pg.display.set_mode((width*2, height*2))

running = True
tau = 2*np.pi

function_drawn = False

f = lambda t: t**3
g = lambda T: T - 1/(27*T)
h = lambda t: t - 1/(3*t)
p = lambda z: z**3 + z

def complex2vector(z):
    return np.array([np.real(z), np.imag(z)])

def vector2complex(v):
    return v[0] + 1j*v[1]

def preimages(func, c, start_values):
    f_as_vector = lambda v: complex2vector(func(vector2complex(v)) - c)
    new_values_as_vector = [root(f_as_vector, complex2vector(z0))['x'] for z0 in start_values]
    return [vector2complex(v) for v in new_values_as_vector]




def draw_function(f, in_quadrant = (0,0)):

    x = np.linspace(xmin, xmax, num=dim)
    y = np.linspace(ymin, ymax, num=dim)
    xx,yy = np.meshgrid(x,y)
    z = xx + 1j*yy
    values = f(z)

    HSV[:,:,0] = np.angle(values)/(2*np.pi) + 0.5
    r2 = np.real(values)**2 + np.imag(values)**2 + 1e-10
    l = np.log(r2) * 0.5
    m = l - np.floor(l)
    smear = 0.02
    invsmear = 1/smear
    mm = (m < 1 - smear) * m + (m >= 1 - smear) * (-invsmear * m + invsmear)
    HSV[:,:,2] = 1 - 0.25 * mm
    

    RGB = np.rot90(np.rint(hsv_to_rgb(HSV) * 255),3)
    # array is incorrectly oriented for some reason, need to rotate
    pg.surfarray.blit_array(mySurf,RGB)

    mySurf_scaled = pg.transform.scale(mySurf,(dim_scaled,dim_scaled))
    window.blit(mySurf_scaled,(in_quadrant[1]*width,in_quadrant[0]*height))

def draw_functions():

    draw_function(lambda t: p(h(t)), in_quadrant = (0,0))
    draw_function(g, in_quadrant = (0,1))
    draw_function(p, in_quadrant = (1,0))
    draw_function(lambda w: w, in_quadrant = (1,1))

e = np.cos(TAU/6) + 1j*np.sin(TAU/6)
preimages_t = [3**(-0.5)*e**i for i in range(6)]
preimages_T = [f(t) for t in preimages_t]
preimages_z = [h(t) for t in preimages_t]
preimages_w = [g(f(t)) for t in preimages_t]


def update_preimages(w0 = 0, from_quadrant = (1,1)):
    global preimages_t, preimages_T, preimages_z, preimages_w
    if from_quadrant == (1,1):
        pass
    elif from_quadrant == (1,0):
        w0 = p(w0)
    elif from_quadrant == (0,1):
        w0 = g(w0)
    elif from_quadrant == (0,0):
        w0 = g(f(w0))
    preimages_t = preimages(lambda t: g(f(t)), w0, preimages_t)
    preimages_T = preimages(g, w0, preimages_T)
    preimages_z = preimages(p, w0, preimages_z)
    preimages_w = 6*[w0]

def draw_preimages():

    draw_points(preimages_t, in_quadrant = (0,0))
    draw_points(preimages_T, in_quadrant = (0,1))
    draw_points(preimages_z, in_quadrant = (1,0))
    draw_points(preimages_w, in_quadrant = (1,1))

    pg.display.flip()


def draw_points(points, in_quadrant = (0,0)):
    for (i,sister) in enumerate(points):
        a = 1 - i/5
        pg.draw.circle(window, (a*255, a*255, a*255), complex_to_pixel(sister, in_quadrant = in_quadrant), circle_size, 0)
    


def in_window(zz):
    return ((np.real(zz) > xmin) and (np.real(zz) < xmax) and (np.imag(zz) > ymin) and (np.imag(zz) < ymax))

def abs2(zz):
    return np.real(zz)**2 + np.imag(zz)**2



def pixel_to_complex(x,y, in_quadrant = (0,0)):
    real = xmin + ((float(x) - width*in_quadrant[1])/width) * (xmax - xmin)
    imag = ymax - ((float(y) - height*in_quadrant[0])/height) * (ymax - ymin)
    return real + 1j*imag


def complex_to_pixel(z0, in_quadrant = (0,0)):
    x0 = ((np.real(z0) - xmin)/(xmax - xmin) + in_quadrant[1]) * width
    y0 = (1 - (np.imag(z0) - ymin)/(ymax - ymin) + in_quadrant[0]) * height
    return (int(x0), int(y0))



draw_functions()
draw_preimages()


dragging = False

while running:
    for myEvent in pg.event.get():
        if myEvent.type == pg.QUIT:
            running = False
        elif myEvent.type == pg.MOUSEBUTTONDOWN:
            # start dragging
            dragging = True
            
        elif myEvent.type == pg.MOUSEMOTION and dragging:
            # dragging
            (mouseX,mouseY) = pg.mouse.get_pos()
            quadrant = (int(mouseY > height), int(mouseX > width))
            mouseZ = pixel_to_complex(mouseX,mouseY, in_quadrant = quadrant)
            update_preimages(mouseZ, from_quadrant = quadrant)
            draw_preimages()

        elif myEvent.type == pg.MOUSEBUTTONUP:
            # end dragging
            dragging = False

        elif myEvent.type == pg.KEYDOWN:
            draw_functions()
            draw_preimages()

