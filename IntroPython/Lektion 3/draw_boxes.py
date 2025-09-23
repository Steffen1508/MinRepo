import pygame
import math

pygame.init()
screen = pygame.display.set_mode((640, 480))
screen.fill((255, 255, 255))

start_point = (25, 25)
box_size = (25, 25)

for i in range(0, 10):
    box_position = (start_point[0] + i * box_size[0]*2,
                    start_point[1])
    rect = box_position + box_size
    pygame.draw.rect(screen, (0,0,0), rect)

pygame.display.flip()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

