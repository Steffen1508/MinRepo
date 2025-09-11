import pygame
import math


pygame.init() # Initialize Pygame
screen = pygame.display.set_mode((640, 480)) # Create a window of 640x480 pixels
screen.fill((0, 0, 0)) # Fill the screen with white

length = 200
start_point = [320,240]

pygame.draw.circle(screen, (255,255,255), start_point, 201, 1)


for angle in range(0,360,30):

    x_cord = start_point[0] + length * (math.cos(math.radians(angle)))
    y_cord = start_point[1] + length * (math.sin(math.radians(angle)))

    end_point = [x_cord,y_cord]
    pygame.draw.line(screen, (255, 255, 255), (start_point), (end_point))


length = 170

for angle in range(0,360,30):

    x_cord = start_point[0] + length * (math.cos(math.radians(angle)))
    y_cord = start_point[1] + length * (math.sin(math.radians(angle)))

    end_point = [x_cord,y_cord]
    pygame.draw.line(screen, (0, 0, 0), (start_point), (end_point),5)







# Make sure the window stays open until the user closes it
run_flag = True
while run_flag is True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_flag = False
    pygame.display.flip() # Refresh the screen so drawing appears