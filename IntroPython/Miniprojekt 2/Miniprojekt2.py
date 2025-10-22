import pygame
from perlin_noise import PerlinNoise
import math
import random
import time
pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()
 
screen.fill((255, 255, 255))
terrain_size = 10
grid_width = 640 // terrain_size
grid_height = 480 // terrain_size
map_seed = random.randint(0,100)
terrain=[]
rect_size = (10,10)
points=[]
 
grid_x_start, grid_y_start, grid_x_goal, grid_y_goal = 0,0,0,0
 
noise = PerlinNoise(octaves=2, seed=map_seed)


 
for y in range(grid_height):
    row = []
    terrain.append(row)
    for x in range(grid_width):
        cell = {
        "type": None,
        "color": None,
        "cost": None
}
        randomizer = noise([x/20, y/20]) # Noise for cellen
        if randomizer < -0.2:
            cell["type"] = "water"
            cell["cost"] = float("inf")
            color = (0, 0, 255) # Vand
        
        elif randomizer < 0.2:
            cell["type"] = "grass"
            cell["cost"] = 1
            color = (0, 255, 0) # Græs
        else:
            color = (100, 100, 100) # Bjerg
            cell["type"] = "mountain"
            cell["cost"] = 3
 
        cell["color"] = color
        row.append(cell)
            
        pygame.draw.rect(screen, color,(x * terrain_size, y * terrain_size, terrain_size, terrain_size))
 
 # print(f"Terrain size: {len(terrain)}x{len(terrain[0])}")
 
 
 
while True:
 
    if len(points) == 0:
        start_cord = ()
        goal_cord = ()
    
    if len(points) == 1:
        x, y = points[0]
        rect_start = x, y, rect_size[0], rect_size[1]
        x_start, y_start = points[0]
        grid_x_start = x_start // terrain_size 
        grid_y_start = y_start // terrain_size
        pygame.draw.rect(screen, (0,120,120), rect_start)
    
    elif len(points) == 2:
        x, y = points[1]
        rect_stop = x, y, rect_size[0], rect_size[1]
        x_goal, y_goal = points[1]
        grid_x_goal = x_goal // terrain_size
        grid_y_goal = y_goal // terrain_size
        pygame.draw.rect(screen, (255,0,0), rect_stop)
    
    
    start_cord = grid_x_start, grid_y_start
    goal_cord = grid_x_goal, grid_y_goal
    
    clock.tick(120)
    pygame.display.flip() 
    for event in pygame.event.get():            

        if event.type == pygame.QUIT:
            pygame.quit()
            exit() 
        elif event.type == pygame.MOUSEBUTTONDOWN:
            points.append(pygame.mouse.get_pos())
    
            start_type = terrain[grid_y_start][grid_x_start]["type"]
            goal_type = terrain[grid_y_goal][grid_x_goal]["type"]

    
            if start_type == "water" or goal_type == "water":
                print("Start eller slut er i vand – programmet lukker...")
                time.sleep(5)
                pygame.quit()