import pygame
import random

def main():
    pygame.init()
    screen_size = (640,480)
    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()

    grid = empty_cell((64,48))


    grid_line(grid,(0,0),(63,0))      # top
    grid_line(grid,(0,47),(63,47))    # bottom
    grid_line(grid,(0,0),(0,47))      # left
    grid_line(grid,(63,0),(63,47))    # right



    # --- Main loop ---
    while True:
        screen.fill((255,255,255))
        draw_grid(screen,grid)
        clock.tick(60)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


def empty_cell(size):
    grid=[]
    for y in range(size[1]):
        row = []
        for x in range(size[0]):
            row.append(0)
        grid.append(row)
    return grid


def grid_line(grid,start_cord,end_cord):
    if start_cord[0] != end_cord[0] and start_cord[1] != end_cord[1]:
        print("Fejl, der bliver tegnet en skrå linje")
        pygame.quit()
        exit()
    else:
        if start_cord[0] == end_cord[0]:
            for y in range(start_cord[1], end_cord[1] + 1):
                grid[y][start_cord[0]] = 1
        else:
            for x in range(start_cord[0], end_cord[0] + 1):
                grid[start_cord[1]][x] = 1
        return grid


def draw_grid(screen,grid):
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            if grid[y][x] == 1:
                pygame.draw.rect(screen,(0,0,0),(x*10,y*10,10,10))


if __name__ == "__main__":
    main()
