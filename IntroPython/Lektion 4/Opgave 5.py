import pygame, sys
import math
import numpy as np

pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()




x = 320
y = 480 // 3
radius = 10
tid = 0

screen.fill((255, 255, 255))

while True:



    clock.tick(120)
    pygame.display.flip()    
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos=pygame.mouse.get_pos()
            btn=pygame.mouse
            pygame.draw.circle(screen,(0,0,0), (pos[0],pos[1]), radius)
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()  

    