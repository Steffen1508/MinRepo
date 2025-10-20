#Opgave 7

import pygame, sys
import math
import numpy as np
import random

pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()


start_x = 320
start_y = 240
size = (20,20)
direction_x = 0
direction_y = 0
speed = 1
score = 0

gold =[]


for i in range(10):
    gold.append({
        "id": i,
        "x": random.randint(0, 620),
        "y": random.randint(0, 460),
        "points": random.randint(1, 100)
    })


while True:   

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()  

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                direction_x = 1
                direction_y = 0

            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                direction_x = -1
                direction_y = 0

            elif event.key == pygame.K_UP or event.key == pygame.K_w:
                direction_x = 0
                direction_y = -1

            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                direction_x = 0
                direction_y = 1
        
    if direction_x == 1:
        start_x = start_x + speed
    elif direction_x == -1:
        start_x = start_x - speed
    elif direction_y == 1:
        start_y = start_y + speed
    elif direction_y == -1:
        start_y = start_y - speed


    if start_x == 0:
        start_x = 640
    elif start_x == 640:
        start_x = 0 

    if start_y == 0:
        start_y = 480
    elif start_y == 480:
        start_y = 0

    screen.fill((255, 255, 255))
    rect = start_x,start_y,size[0],size[1]

    pygame.draw.rect(screen, (0,0,0), rect)
    squre_rect = pygame.Rect(start_x, start_y, size[0], size[1])

    new_gold = []

    for g in gold:
        pygame.draw.circle(screen, (255, 215, 0), (g["x"], g["y"]), g["points"])
        font = pygame.font.Font('freesansbold.ttf', g["points"])
        text = font.render(str(g["points"]), True, (0,0,0))
        textRect = text.get_rect()
        textRect.center = (g["x"],g["y"] )
        screen.blit(text, textRect)
        
        # Beregn midtpunkterne RENT CHATGPT KODE
        player_center = pygame.math.Vector2(start_x + size[0] / 2, start_y + size[1] / 2)
        gold_center = pygame.math.Vector2(g["x"], g["y"])

        # Beregn afstanden mellem centrum af spiller og guld
        distance = player_center.distance_to(gold_center)

        # Sammenlign med guldets radius + spillerens "halv størrelse"
        if distance < g["points"] + size[0] / 2:
            score += g["points"]
            print(f"Total score: {score}") #HER SLUTTER CHATGPT KODE
        else:
            new_gold.append(g)

    gold = new_gold

    font = pygame.font.Font('freesansbold.ttf', 20)

    score_text = font.render(f"Score: {score}", True, (0,0,0))
    screen.blit(score_text, (10, 10))

    if len(gold) == 0:
        for i in range(10):
            gold.append({
                "id": i,
                "x": random.randint(0, 620),
                "y": random.randint(0, 460),
                "points": random.randint(1, 100)
            })


    clock.tick(120)
    pygame.display.flip() 

    