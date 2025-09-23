import pygame
import math

pygame.init()
screen = pygame.display.set_mode((640, 480))
screen.fill((255, 255, 255))

box_position = (0, 0)
box_size = 1
for i in range (0,100):

    box_position = (box_position[0] + box_size, box_position[1] + box_size)
    box_size = box_size + 1
    rect = box_position + (box_size, box_size)




    pygame.draw.rect(screen, (0,0,0), rect)


pygame.display.flip()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

