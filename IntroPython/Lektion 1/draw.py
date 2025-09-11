import pygame

pygame.init() # Initialize Pygame
screen = pygame.display.set_mode((640, 480)) # Create a window of 640x480 pixels
screen.fill((255, 255, 255)) # Fill the screen with white




# Draw the ground
pygame.draw.line(screen, (0, 255, 0), (0, 380), (1000, 380)) # Draw a green line

# Draw the bottom of the house
pygame.draw.line(screen, (0, 0, 0), (100, 375), (540, 375)) # Draw a black line

# Draw two walls
pygame.draw.line(screen, (0, 0, 0), (100, 375), (100, 100)) # Draw a black line
pygame.draw.line(screen, (0, 0, 0), (540, 375), (540, 100)) # Draw a black line

# Draw the roof
pygame.draw.line(screen, (0, 0, 0), (100, 100), (320, 40)) # Draw a black line
pygame.draw.line(screen, (0, 0, 0), (320, 40), (540, 100)) # Draw a black line
# Make sure the window stays open until the user closes it
run_flag = True
while run_flag is True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_flag = False
    pygame.display.flip() # Refresh the screen so drawing appears