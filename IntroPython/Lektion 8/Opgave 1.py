import pygame
import random
import math

class Data_point:
    def __init__(self, position, lable, color):
        self.position = position
        self.lable = lable
        self.color = color
    
    def draw(self,screen):
        pygame.draw.circle(screen,self.color,self.position,5)


class kNN:
    def __init__(self,k):
        self.k = k
    
    def cal_dist(self,p1,p2):
        x = p1[0]-p2[0]
        y = p1[1]-p2[1]
        distance = math.sqrt(x**2+y**2)
        return distance
    

    def predict(self, sample, data_points):
        distances = []
        for point in data_points:
            dist = self.cal_dist(sample.position, point.position)
            distances.append((dist, point.lable))


        distances.sort(key=lambda x: x[0])

        neighbors = distances[:self.k]
        
        counts = {}
        for dist, lable in neighbors:
            counts[lable] = counts.get(lable,0) + 1

        best_lable = max(counts, key=counts.get)
        return best_lable
    
    
    



def main():
    screen = initialize_pygame()

    data_points = []
    for n in range(10):
        sample = Data_point((random.gauss(200,20), random.gauss(200,20)),"salmon",(255,0,0))
        data_points.append(sample)
    
    for n in range(10):
        sample = Data_point((random.gauss(400,20), random.gauss(200,20)),"sea bass",(0,0,255))
        data_points.append(sample)

    for data_point in data_points:
        data_point.draw(screen)

    sample = Data_point((random.gauss(100,500), random.gauss(100,500)),"unknown",(0,0,0))

    sample.draw(screen)

    knn = kNN(3)
    predicted_lable = knn.predict(sample, data_points)
    print(predicted_lable)


    if predicted_lable == "salmon":
        sample.color = (255,0,0)
    else:
        sample.color = (0,0,255)
    sample.draw(screen)


   
    game_loop()




def initialize_pygame():
    pygame.init()
    screen_width = 640
    screen_height = 480
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pygame")
    screen.fill((255, 255, 255))
    return screen


def game_loop():
    
    clock = pygame.time.Clock()
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()

        clock.tick(60)

if __name__ == "__main__":
    main()