#!/usr/bin/env python2.7

import pygame as pg
import math
import numpy as np
import numpy.polynomial.polynomial as npp
import numdifftools as nd
from matplotlib.colors import hsv_to_rgb

dim = 301
scale = 2
dim_scaled = scale * dim
circle_size = 10
font_size = 30
rootsToggled = False

values = np.ndarray((dim,dim))
HSV = RGB = np.ndarray((dim,dim,3))
HSV[:,:,1] = 1
#RGB_scaled = np.ndarray((2*dim,2*dim,3))
mySurf = pg.Surface((dim,dim))
mySurf_scaled = pg.Surface((dim_scaled,dim_scaled))

pg.init()
myfont = pg.font.SysFont(None, font_size)


xmin = ymin = -5
xmax = ymax = 5

f = (lambda t: t)

x = np.linspace(xmin, xmax, num=dim)
y = np.linspace(ymin, ymax, num=dim)
xx,yy = np.meshgrid(x,y)
z = xx + 1j*yy



(width, height) = (dim_scaled,dim_scaled)
window = pg.display.set_mode((width, height))

running = True
mouseDown = False
myRoots = []
myRootValues = []
f_as_poly = npp.Polynomial(1)
values = np.ones(np.shape(z),dtype=np.complex128)

intro_text = 'Click anywhere to create a draggable root.\nToggle root values with SPACE.\nClear the graph with ESC.'
introtextsurface = myfont.render(intro_text, True, (0, 0, 0))

introtext_x_pos = 20
introtext_y_pos = 20


rootsurface = myfont.render('', True, (0, 0, 0))

def render_multi_line(text, x, y, fsize):
    lines = text.splitlines()
    for i, l in enumerate(lines):
        window.blit(myfont.render(l, True, (0,0,0)), (x, y + fsize*i))


def rootText():
    text = "Roots:\r\n"
    for root in myRoots:
        text += str(round(root.x,2))
        if root.y >= 0:
            text += " + " + str(round(root.y,2)) + " i"
        else:
            text += " - " + str(-round(root.y,2)) + " i"
        text += "\r\n"
    return text



def redraw():
    global values
    values = np.ones(np.shape(z),dtype=np.complex128)
    derivative = np.zeros(np.shape(z),dtype=np.complex128)
    for root in myRoots:
        zprime = z - (root.x + root.y *1j)
        values *= zprime
        derivative += 1/zprime
    
    derivative *= values
    
    HSV[:,:,0] = np.angle(values)/(2*np.pi) + 0.5
    r2 = np.real(values)**2 + np.imag(values)**2
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
    #pg.surfarray.blit_array(mySurf,RGB)
    #window.blit(mySurf,(0,0))
    
    drawCriticalLines2()
    
    for root in myRoots:
        root.display()
    
    if len(myRoots) == 0:
        render_multi_line(intro_text,introtext_x_pos, introtext_y_pos,font_size)

    if rootsToggled:
        #rootsurface = myfont.render(rootText(), True, (0, 0, 0))
        render_multi_line(rootText(),width - 120,5,font_size)

    pg.display.flip()



def drawCriticalLines1():

    # draw critical lines
    eps1 = 1e0
    eps2 = 2e-3
    
    isCritical = (np.abs(np.real(derivative)) < eps1) * (np.abs(np.imag(derivative)) < eps1)
    isNotCritical = 1 - isCritical
    criticalHues = HSV[:,:,0] * isCritical
    
    for hue in np.nditer(criticalHues):
        if hue != 0:
            preimageMask = (np.abs(HSV[:,:,0] - hue) < eps2)
            HSV[:,:,2] *= 1 - preimageMask


def inWindow(zz):
    return ((np.real(zz) > xmin) and (np.real(zz) < xmax) and (np.imag(zz) > ymin) and (np.imag(zz) < ymax))


def mod_gradient(zz): # gradient direction of the modulus (as complex number)
    return 2*np.sum(1/np.conj(zz-myRootValues))

def abs2(zz):
        return np.real(zz)**2 + np.imag(zz)**2



def draw_contour_from(z0,eps):

    
    counter = 0
    towardsZero = False
    eps4 = 1e-1
    
    while inWindow(z0) and counter < 1e3 and abs2(npp.polyval(z0,f_as_poly)) > eps4:
        
        dz = eps * mod_gradient(z0)/np.abs(mod_gradient(z0))

        color = (0,0,0)
        if eps > 0:
            color = (255,255,255)

        pg.draw.line(window, color, complex_to_pixel(z0), complex_to_pixel(z0+dz),3)
        
        if counter == 0:
            if towards_zero(z0,dz):
                towardsZero = True
        else:
            if towardsZero and not towards_zero(z0,dz):
                break
        
        z0 += dz
        counter += 1


def towards_zero(z0,dz):
    return abs2(npp.polyval(z0+dz,f_as_poly)) < abs2(npp.polyval(z0,f_as_poly))

def drawCriticalLines2():

    eps3 = 1e-1
    
    derivative = npp.polyder(f_as_poly)
    criticalPoints = npp.polyroots(derivative)
    
    #phase = (lambda v: np.angle(npp.polyval(v[0]+1j*v[1],f_as_poly)))
    phase = np.angle(values)
    #hessian = nd.Hessian(phase)
    
    
    for z0 in criticalPoints:
        
        z1 = z0 + eps3
        z2 = z0 + eps3
        z3 = z0 - eps3
        z4 = z0 - eps3

        draw_contour_from(z1,eps3)
        draw_contour_from(z2,-eps3)
        draw_contour_from(z3,eps3)
        draw_contour_from(z4,-eps3)

        (x0, y0) = complex_to_pixel(z0)
        pg.draw.circle(window, (255,0,0), (x0,y0),circle_size,0)
        pg.draw.circle(window, (0,0,0), (x0,y0),circle_size,3)




def pixel_to_complex(x,y):
    real = xmin + float(x)/width * (xmax - xmin)
    imag = ymax - float(y)/height * (ymax - ymin)
    return real + 1j*imag


def complex_to_pixel(z0):
    x0 = (np.real(z0) - xmin)/(xmax - xmin) * width
    y0 = height - (np.imag(z0) - ymin)/(ymax - ymin) * height
    return (int(x0), int(y0))



class Root:
    
    def __init__(self, z0, size):
        self.x = np.real(z0)
        self.y = np.imag(z0)
        self.size = size
        self.colour = (127,127,127)
        self.thickness = 3
    
    def display(self):
        pg.draw.circle(window, self.colour, (self.pixel_coord_x(), self.pixel_coord_y()), self.size, 0)
        pg.draw.circle(window, (0,0,0), (self.pixel_coord_x(), self.pixel_coord_y()), self.size, self.thickness)

    def pixel_coord_x(self):
        return int((self.x - xmin)/(xmax - xmin) * width)

    def pixel_coord_y(self):
        return int(height - (self.y - ymin)/(ymax - ymin) * height)

    def value(self):
        return self.x + 1j*self.y




def findRoot(mouseZ):
    for p in myRoots:
        if (p.x-np.real(mouseZ))**2 + (p.y-np.imag(mouseZ))**2 <= 0.3**2:
            return p
    return None

redraw()

while running:
    for myEvent in pg.event.get():
        if myEvent.type == pg.QUIT:
            running = False
        elif myEvent.type == pg.MOUSEBUTTONDOWN:
            mouseDown = True
            # create a new zero or start dragging one
            (mouseX,mouseY) = pg.mouse.get_pos()
            mouseZ = pixel_to_complex(mouseX,mouseY)
            selectedRoot = findRoot(mouseZ)
            if selectedRoot == None:
                selectedRoot = Root(mouseZ,circle_size)
                myRoots.append(selectedRoot)
                myRootValues.append(selectedRoot.value())
                f_as_poly = npp.polyfromroots(myRootValues)
                selectedRoot.display()
                redraw()
                
        elif myEvent.type == pg.MOUSEMOTION and mouseDown:
            # dragging
            (mouseX,mouseY) = pg.mouse.get_pos()
            mouseZ = pixel_to_complex(mouseX,mouseY)
            selectedRoot.x = np.real(mouseZ)
            selectedRoot.y = np.imag(mouseZ)
            myRootValues = []
            for root in myRoots:
                myRootValues.append(root.value())
            f_as_poly = npp.polyfromroots(myRootValues)
            redraw()

        elif myEvent.type == pg.MOUSEBUTTONUP:
            mouseDown = False
            selectedRoot = None

        elif myEvent.type == pg.KEYDOWN:
            pressed = pg.key.get_pressed()
            if pressed[pg.K_SPACE]:
                rootsToggled = not(rootsToggled)
                redraw()
            elif pressed[pg.K_ESCAPE]:
                myRoots = []
                myRootValues = []
                f_as_poly = npp.Polynomial(1)
                criticalPoints = []
                redraw()





















