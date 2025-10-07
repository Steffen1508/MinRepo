import pygame
import math
from datetime import datetime


pygame.mixer.pre_init(44100, -16, 2, 256)  # mindre buffer => lavere latency (chatgpt hjælp)
pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

start_point = [320, 240]
viser_length_sekund = 165
viser_length_minut = 150
viser_length_timer = 110
viser_start_point = [320, 240]

pygame.mixer.init()
# 01.mp3 ... 60.mp3  -> giver 60 elementer (index 0..59)
sounds = [pygame.mixer.Sound(f"sounds/{i:02d}.mp3") for i in range(1, 61)]

last_second = datetime.now().second
run_flag = True

while run_flag:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_flag = False

    # === Brug samme timestamp til ALT i dette frame ===
    now = datetime.now()
    seconds = now.second
    minutes = now.minute
    hours   = now.hour

    # Baggrund (gradient)
    for y in range(480):
        farve = (int(20 + 80 * y / 480), int(30 + 60 * y / 480), int(100 + 100 * y / 480))
        pygame.draw.line(screen, farve, (0, y), (640, y))

    # Ramme
    pygame.draw.circle(screen, (255, 255, 255), start_point, 201, 2)
    pygame.draw.circle(screen, (40, 40, 60), start_point, 210, 5)

    # Sekund-takker (prikker)
    for angle in range(0, 360, 6):
        x_outer = start_point[0] + 200 * math.cos(math.radians(angle))
        y_outer = start_point[1] + 200 * math.sin(math.radians(angle))
        pygame.draw.circle(screen, (150, 150, 200), (int(x_outer), int(y_outer)), 1)

    # Minut-takker (tykke)
    for angle in range(0, 360, 30):
        x_outer = start_point[0] + 200 * math.cos(math.radians(angle))
        y_outer = start_point[1] + 200 * math.sin(math.radians(angle))
        x_inner = start_point[0] + 180 * math.cos(math.radians(angle))
        y_inner = start_point[1] + 180 * math.sin(math.radians(angle))
        pygame.draw.line(screen, (255, 255, 255), (x_inner, y_inner), (x_outer, y_outer), 3)

    # Visere
    grad_sek = seconds * 6 - 90
    x = viser_start_point[0] + viser_length_sekund * math.cos(math.radians(grad_sek))
    y = viser_start_point[1] + viser_length_sekund * math.sin(math.radians(grad_sek))
    pygame.draw.line(screen, (255, 80, 80), viser_start_point, (x, y), 2)

    grad_min = minutes * 6 - 90
    x = viser_start_point[0] + viser_length_minut * math.cos(math.radians(grad_min))
    y = viser_start_point[1] + viser_length_minut * math.sin(math.radians(grad_min))
    pygame.draw.line(screen, (0, 200, 255), viser_start_point, (x, y), 4)

    grad_timer = ((hours % 12) * 30) + (minutes * 0.5) - 90
    x = viser_start_point[0] + viser_length_timer * math.cos(math.radians(grad_timer))
    y = viser_start_point[1] + viser_length_timer * math.sin(math.radians(grad_timer))
    pygame.draw.line(screen, (255, 180, 0), viser_start_point, (x, y), 6)

    pygame.draw.circle(screen, (255, 255, 255), start_point, 6)

    # Digital tid (brug samme 'now')
    font = pygame.font.SysFont("Consolas", 32, bold=True)
    time_str = now.strftime("%H:%M:%S")
    text = font.render(time_str, True, (255, 255, 255))
    shadow = font.render(time_str, True, (50, 50, 50))
    rect = text.get_rect(center=(320, 400))
    shadow_rect = rect.copy(); shadow_rect.move_ip(3, 3)
    screen.blit(shadow, shadow_rect)
    screen.blit(text, rect)

    test = [4,9,14,19,24,29,34,39,44,49,54,59]
    # Lyd KUN når sekundet skifter, og kun hvert 5. sekund
    if seconds != last_second:
        if seconds in test:
            # seconds går 0..59 -> maps til index 0..59 (01.mp3..60.mp3)
            sounds[seconds].play()
        last_second = seconds

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
