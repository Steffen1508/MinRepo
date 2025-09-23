import pygame
import math

pygame.init()
screen = pygame.display.set_mode((640, 480))
screen.fill((255, 255, 255))

start_point = (25, 25)
box_size = (50, 50)
colors = [(255, 255, 255), (0, 0, 0)]  # hvid og sort


for r in range (0,400,50):
    for c in range (0,400,50):
        x = start_point[0] + c
        y = start_point[1] + r

        col = x // 50 
        row = y // 50
        idx = ((col ^ row) & 1)

        rect = (x, y, box_size[0], box_size[1])
        color = colors[idx]
        pygame.draw.rect(screen, color, rect)


pygame.display.flip()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

