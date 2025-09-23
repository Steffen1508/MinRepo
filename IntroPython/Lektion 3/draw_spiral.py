import pygame
import math

pygame.init()
screen = pygame.display.set_mode((640, 480))
screen.fill((255, 255, 255))

screen_size = (640, 480)

start_position = (screen_size[0]/2, screen_size[1]/2)
box_size = (5, 5)
offset = 10

for i in range (1, 360*5, 5):

    r = 0.1*i
    x = r*math.cos(math.radians(i))
    y = r*math.sin(math.radians(i))

    box_pos = (start_position[0]+x,start_position[1]+y)
    rect = (box_pos, box_size)


    pygame.draw.rect(screen, (0,0,0), rect)


pygame.display.flip()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

