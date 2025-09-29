import pygame
import math

pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

hastighed = 1
i=0

start_point_x = 320
start_point_y = 240
sphere_radius = 10

while True:
    screen.fill((255, 255, 255))
    
    

    start_point_y = start_point_y + hastighed

    if start_point_y >= 480 - sphere_radius:
        start_point_y = 480 - sphere_radius
        hastighed = -hastighed

    elif start_point_y <= sphere_radius:
        start_point_y = sphere_radius
        hastighed = -hastighed
    pygame.draw.circle(screen,(0,0,0), (start_point_x,start_point_y), sphere_radius)


    clock.tick(120)
    pygame.display.flip()    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()  