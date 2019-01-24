#!/usr/bin/env python2.7

import pygame as pg
import math
import numpy as np
import numpy.polynomial.polynomial as npp
import numdifftools as nd
from matplotlib.colors import hsv_to_rgb

dim = 601
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


xmin = ymin = -3
xmax = ymax = 3

f = (lambda t: t)

x = np.linspace(xmin, xmax, num=dim)
y = np.linspace(ymin, ymax, num=dim)
xx,yy = np.meshgrid(x,y)
z = xx + 1j*yy



(width, height) = (dim_scaled,dim_scaled)
window = pg.display.set_mode((width, height))

running = True
mouseDown = False
#initialRoots = [0j,1+0j,1j,-1+0j,-1j]
n = 3
tau = 2*np.pi
e = np.cos(tau/n) + 1j * np.sin(tau/n)
initialRoots = [1j, -1j] # [1+0j, 1.5+0j, (1+0j)*e, (1.5+0j)*e, (1+0j)*e**2, (1.5+0j)*e**2]
nbRoots = len(initialRoots)
myRoots = initialRoots
myRootValues = []
current_value = 0
f_as_poly = npp.polyfromroots(myRoots)
values = np.ones(np.shape(z),dtype=np.complex128)

sister_points = initialRoots
function_drawn = False

def redraw():
    global values, function_drawn

    if not function_drawn:

        # values = np.ones(np.shape(z),dtype=np.complex128)
        # for root in myRoots:
        #     zprime = z - root
        #     values *= zprime
        values = z + 1.0/z
        
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
        window.blit(mySurf_scaled,(0,0))

        function_drawn = True
    
    drawPreimages()

    pg.display.flip()


def drawPreimages():
    global sister_points
    shifter = np.zeros(nbRoots + 1,dtype=np.complex128)
    shifter[0] = current_value
    p = f_as_poly - shifter
    new_sister_points = npp.polyroots(p)
    ordered_new_sister_points = []
    used_indices = []
    for sis in sister_points:
        quadrances = np.array([abs2(sis - new_sis) for new_sis in new_sister_points])
        for i in used_indices:
            np.delete(quadrances,i)
        i = np.argmin(quadrances)
        used_indices.append(i)
        ordered_new_sister_points.append(new_sister_points[i])
    sister_points = ordered_new_sister_points
    for (i,sister) in enumerate(sister_points):
        #pg.draw.circle(window, (0,0,0), complex_to_pixel(sister), circle_size + 2, 0)
        a = 1-i/(nbRoots-1)
        pg.draw.circle(window, (a*255, a*255, a*255), complex_to_pixel(sister), circle_size, 0)
    


def inWindow(zz):
    return ((np.real(zz) > xmin) and (np.real(zz) < xmax) and (np.imag(zz) > ymin) and (np.imag(zz) < ymax))


def mod_gradient(zz): # gradient direction of the modulus (as complex number)
    return 2*np.sum(1/np.conj(zz-myRootValues))

def abs2(zz):
        return np.real(zz)**2 + np.imag(zz)**2



def pixel_to_complex(x,y):
    real = xmin + float(x)/width * (xmax - xmin)
    imag = ymax - float(y)/height * (ymax - ymin)
    return real + 1j*imag


def complex_to_pixel(z0):
    x0 = (np.real(z0) - xmin)/(xmax - xmin) * width
    y0 = height - (np.imag(z0) - ymin)/(ymax - ymin) * height
    return (int(x0), int(y0))





def findSister(mouseZ):
    for p in sister_points:
        if (np.real(p)-np.real(mouseZ))**2 + (np.imag(p)-np.imag(mouseZ))**2 <= 0.3**2:
            return p
    return None

def foundSister(mouseZ):
    return (findSister(mouseZ) != None)


redraw()

dragging = False

while running:
    for myEvent in pg.event.get():
        if myEvent.type == pg.QUIT:
            running = False
        elif myEvent.type == pg.MOUSEBUTTONDOWN:
            # # create a new zero or start dragging one
            (mouseX,mouseY) = pg.mouse.get_pos()
            mouseZ = pixel_to_complex(mouseX,mouseY)
            dragging = foundSister(mouseZ)
            
        elif myEvent.type == pg.MOUSEMOTION and dragging:
            # dragging
            (mouseX,mouseY) = pg.mouse.get_pos()
            handle = pixel_to_complex(mouseX,mouseY)
            current_value = npp.polyval(handle, f_as_poly)
            redraw()

        elif myEvent.type == pg.MOUSEBUTTONUP:
            dragging = False



