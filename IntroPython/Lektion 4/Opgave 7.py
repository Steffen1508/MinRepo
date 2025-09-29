#Opgave 7

import pygame, sys
import math
import numpy as np

pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()



points=[]

screen.fill((255, 255, 255))

while True:

    if len(points) > 1:
        pygame.draw.lines(screen, (0,0,0), False, points, width=3)

    clock.tick(120)
    pygame.display.flip()    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()  


        elif event.type == pygame.MOUSEBUTTONDOWN:
            points.append(pygame.mouse.get_pos())
            pos_now = pygame.mouse.get_pos()
            print(points)
            print(pos_now)

            x,y = points[0]

            pos_x = pos_now[0] - x
            pos_y = pos_now[1] - y


    if len(points) > 1 and pos_y <= 5 and pos_x <= 5:
        pygame.draw.polygon(screen,(255,0,0),points)
        points=[]


    