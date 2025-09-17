import pygame
import math
import time
from datetime import datetime

pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

start_time = time.perf_counter()
length = 200
start_point = [320, 240]
viser_length_sekund = 165
viser_length_minut = 150
viser_length_timer = 110
viser_start_point = [320, 240]

pygame.mixer.init()
sounds = [pygame.mixer.Sound(f"sounds/{i:02d}.mp3") for i in range(1, 61)]

def draw_digital_clock(screen):
    font = pygame.font.SysFont("Consolas", 24, bold=True)

    box_rect = pygame.Rect(0, 0, 1100, 850)

    time_str = datetime.now().strftime("%H:%M:%S")
    text_surf = font.render(time_str, True, (80, 255, 120))
    text_rect = text_surf.get_rect(center=box_rect.center)
    screen.blit(text_surf, text_rect)

last_second = datetime.now().second

run_flag = True
while run_flag:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_flag = False

    # screen
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (255, 255, 255), start_point, 201, 1)

    # Sekund takker
    length = 200
    for angle in range(0, 360, 6):
        x_cord = start_point[0] + length * math.cos(math.radians(angle))
        y_cord = start_point[1] + length * math.sin(math.radians(angle))
        end_point = [x_cord, y_cord]
        pygame.draw.line(screen, (255, 255, 255), start_point, end_point)

    length = 190
    for angle in range(0, 360, 6):
        x_cord = start_point[0] + length * math.cos(math.radians(angle))
        y_cord = start_point[1] + length * math.sin(math.radians(angle))
        end_point = [x_cord, y_cord]
        pygame.draw.line(screen, (0, 0, 0), start_point, end_point, 5)

    # Minut takker
    length = 200
    for angle in range(0, 360, 30):
        x_cord = start_point[0] + length * math.cos(math.radians(angle))
        y_cord = start_point[1] + length * math.sin(math.radians(angle))
        end_point = [x_cord, y_cord]
        pygame.draw.line(screen, (255, 255, 255), start_point, end_point)

    length = 170
    for angle in range(0, 360, 30):
        x_cord = start_point[0] + length * math.cos(math.radians(angle))
        y_cord = start_point[1] + length * math.sin(math.radians(angle))
        end_point = [x_cord, y_cord]
        pygame.draw.line(screen, (0, 0, 0), start_point, end_point, 5)

    now = datetime.now()
    seconds = now.second
    minutes = now.minute
    hours = now.hour

    # Viser
    grad_sek = seconds * 6 - 90
    viser_sekund_x = viser_start_point[0] + viser_length_sekund * math.cos(math.radians(grad_sek))
    viser_sekund_y = viser_start_point[1] + viser_length_sekund * math.sin(math.radians(grad_sek))
    pygame.draw.aaline(screen, (255, 0, 0), viser_start_point, (viser_sekund_x, viser_sekund_y))

    grad_min = minutes * 6 - 90
    viser_minut_x = viser_start_point[0] + viser_length_minut * math.cos(math.radians(grad_min))
    viser_minut_y = viser_start_point[1] + viser_length_minut * math.sin(math.radians(grad_min))
    pygame.draw.aaline(screen, (255, 0, 0), viser_start_point, (viser_minut_x, viser_minut_y))

    grad_timer = ((hours % 12) * 30) + (minutes * 0.5) - 90
    viser_timer_x = viser_start_point[0] + viser_length_timer * math.cos(math.radians(grad_timer))
    viser_timer_y = viser_start_point[1] + viser_length_timer * math.sin(math.radians(grad_timer))
    pygame.draw.aaline(screen, (255, 0, 0), viser_start_point, (viser_timer_x, viser_timer_y))

    pygame.draw.circle(screen, (255, 255, 255), start_point, 5)

    # Lyd pr. sekund
    sec = now.second
    if sec != last_second:
        sounds[sec].play()
        last_second = sec

    # >>> TEGN DIGITALT UR HER <<<
    draw_digital_clock(screen)

    # Flip + tick
    pygame.display.flip()
    clock.tick(120)

pygame.quit()
