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

    # --- Labyrint ---
    make_maze(grid)
    print(grid)


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


def make_maze(grid):
    # Lodrette vægge i hver anden kolonne, hver med et tilfældigt hul,
    # så der altid er en vej igennem labyrinten.
    height = len(grid)
    width = len(grid[0])
    for x in range(2, width - 2, 2):
        gap = random.randint(1, height - 2)       # hullet i væggen
        if gap > 1:
            grid_line(grid, (x, 1), (x, gap - 1))           # væg over hullet
        if gap < height - 2:
            grid_line(grid, (x, gap + 1), (x, height - 2))  # væg under hullet
    return grid


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
# Grov forklaring af hvordan programmet virker:
#
# 1. main() starter pygame og laver et vindue på 640x480.
# 2. empty_cell((64,48)) laver et 2D-grid (liste af lister) med 48 rækker
#    og 64 kolonner, fyldt med 0'er. 0 = tom plads, 1 = væg.
# 3. grid_line(grid, start, slut) sætter 1'ere fra start til slut.
#    Den kan kun tegne vandrette eller lodrette linjer - hvis både x og y
#    er forskellige (skrå linje), giver den en fejl og lukker programmet.
# 4. Først tegnes de fire kanter rundt om hele griddet med grid_line.
# 5. make_maze(grid) laver selve labyrinten:
#    - Den går gennem hver anden kolonne (x = 2, 4, 6, ...).
#    - I hver kolonne vælges et tilfældigt hul (en y-værdi).
#    - Der tegnes en lodret væg over hullet og en under hullet,
#      så der kun er én åbning i hver væg.
#    - Fordi hver væg har et hul, kan man altid komme fra venstre til højre.
# 6. print(grid) printer hele griddet i terminalen.
# 7. I while-løkken tegner draw_grid() hver celle med værdien 1 som en
#    10x10 sort firkant (64*10 = 640 og 48*10 = 480, så det fylder skærmen).
#    Løkken kører 60 gange i sekundet, indtil man lukker vinduet.
# ----------------------------------------------------------------------
