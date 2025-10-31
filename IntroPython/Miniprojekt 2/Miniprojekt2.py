import pygame
from perlin_noise import PerlinNoise
import random
from queue import PriorityQueue
import time
import math

pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

screen.fill((255, 255, 255))
terrain_size = 10
grid_width = 640 // terrain_size
grid_height = 480 // terrain_size
map_seed = random.randint(0,100)
rect_size = (10,10)
points=[]
grid_x_start, grid_y_start, grid_x_goal, grid_y_goal = 0,0,0,0

class grid():
    def __init__(self,width,height,cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.terrain = []

    def generate(self,seed):
        self.terrain = []
        noise = PerlinNoise(octaves=2, seed=seed)
        for y in range(self.height):
            row = []
            for x in range(self.width):
                cell = {"type": None, "color": None, "cost": None}
                randomizer = noise([x / 20, y / 20])
                if randomizer < -0.2:
                    cell["type"] = "water"
                    cell["cost"] = 5
                    cell["color"] = (40, 80, 200)
                elif randomizer < 0.2:
                    cell["type"] = "grass"
                    cell["cost"] = 1
                    cell["color"] = (20, 150, 20)
                else:
                    cell["type"] = "mountain"
                    cell["cost"] = 3
                    cell["color"] = (100, 100, 100)
                row.append(cell)
            self.terrain.append(row)

    def draw(self,screen):
        for y in range(self.height):
            for x in range(self.width):
                color = self.terrain[y][x]["color"]
                rect = ((x * self.cell_size),(y * self.cell_size),self.cell_size,self.cell_size)
                pygame.draw.rect(screen,color,rect)

    def get_neighbors(self,pos):
        x,y = pos
        neighbors = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1,1), (1,-1), (-1,1), (-1,-1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
        return neighbors

    def get_cost(self,pos):
        x,y = pos
        return self.terrain[y][x]["cost"]
    
    def draw_path(self, screen, path):
        if len(path) < 2:
            return
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            start_pos = (
                x1 * self.cell_size + self.cell_size // 2,
                y1 * self.cell_size + self.cell_size // 2
            )
            end_pos = (
                x2 * self.cell_size + self.cell_size // 2,
                y2 * self.cell_size + self.cell_size // 2
            )
            pygame.draw.line(screen, (160, 0, 255), start_pos, end_pos, 4)

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def a_star_algorithm(grid, start, goal, screen):
    frontier = PriorityQueue()
    frontier.put((0, start))  
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current = frontier.get()[1]  
        if current == goal:
            break

        for next in grid.get_neighbors(current):
            new_cost = cost_so_far[current] + grid.get_cost(next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put((priority, next))
                came_from[next] = current

    current = goal
    path = []
    while current != start and current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


my_grid = grid(grid_width,grid_height,terrain_size)
my_grid.generate(map_seed)
my_grid.draw(screen)

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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                map_seed = random.randint(0,100)
                points=[]
                my_grid.generate(map_seed)
                my_grid.draw(screen)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            points.append(pygame.mouse.get_pos())

        start_type = my_grid.terrain[grid_y_start][grid_x_start]["type"]
        goal_type = my_grid.terrain[grid_y_goal][grid_x_goal]["type"]

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                path = a_star_algorithm(my_grid, start_cord, goal_cord, screen)
                my_grid.draw(screen)
                my_grid.draw_path(screen, path)

                pygame.draw.rect(screen, (0,120,120), (x_start, y_start, rect_size[0], rect_size[1]))
                pygame.draw.rect(screen, (255,0,0), (x_goal, y_goal, rect_size[0], rect_size[1]))
                pygame.display.flip()

        if start_type == "water" or goal_type == "water":
            print("Start eller slut er i vand – programmet lukker...")
            time.sleep(5)
            pygame.quit()
            exit()
