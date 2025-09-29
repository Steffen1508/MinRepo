import pygame
import math
import numpy as np

pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()




x = 320
y = 480 // 3
radius = 10
tid = 0

while True:
    screen.fill((255, 255, 255))
    speed = math.sin(tid)
    

    y = y + speed


    print(speed)

    pygame.draw.circle(screen,(0,0,0), (x,y), radius)

    tid += 0.01
    clock.tick(120)
    pygame.display.flip()    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()  