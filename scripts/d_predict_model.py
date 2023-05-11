import pygame
import cv2 as cv
import statistics

from neural_network import *

#Image Constants
WIDTH = 1280
HEIGHT = 720
FPS = 30
UPDATE_RATIO = 2
CAM_PORT = 0
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_BUFFER = 2
TOTAL_VALUE_BUFFER_SIZE = 100
TOLERANCE = 0.001

#Hough Circle Constants
GRADIENT= cv.HOUGH_GRADIENT
DP = 1.05
MINDIST = 70
PARAM1 = 50
PARAM2 = 45
MINRADIUS = 10
MAXRADIUS = 200

#Train Constants
TRAIN_DIR_BOOL = True
TRAIN_DIR = "./neural_network/train_data"
TEST_DIR = "./neural_network/test_data"
MODEL_DIR = "./neural_network/models"
MODEL_NAME = "coin_counter.model"

# Define Colors 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 125, 0)
YELLOW = (255, 225, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

#Other Constants
FONT_SIZE = 18
COIN_TYPES = ['penny', 'nickel', 'dime', 'quarter', 'half_dollar', 'dollar_coin']
COIN_VALUES= [   0.01,     0.05,   0.10,      0.25,          0.50,          1.00]

def main():

    ## initialize pygame and create window
    pygame.init()
    pygame.mixer.init()  ## For sound
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Coin Counter")
    clock = pygame.time.Clock()     ## For syncing the FPS

    ## group all the sprites together for ease of update
    all_sprites = pygame.sprite.Group()

    #Define Camera
    cam = cv.VideoCapture(CAM_PORT, cv.CAP_DSHOW)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    #Define Font
    font = pygame.font.Font('freesansbold.ttf', FONT_SIZE)
    font_title = pygame.font.Font('freesansbold.ttf', FONT_SIZE*2)

    #Define Global Variables
    paused = True
    update = 0
    coin_circles = None
    coin_images_nd = None
    coin_images_py = None
    coin_pred = []
    coin_pred_str = []
    total_value_buffer = []
    w_width = WIDTH
    w_height = HEIGHT

    #Determine confidence colors, bad to good
    confidence = [
        RED,
        ORANGE,
        YELLOW,
        GREEN
    ]

    #Load Model
    model = network_load(f"{MODEL_DIR}/{MODEL_NAME}")

    #Get Names
    coin_types_array = get_coin_names(TRAIN_DIR, COIN_TYPES)

    #Calculate Total Value
    def calc_total_value(coin_str_list: list, coin_types: list, coin_values: list):
        """This function calculates the total value of the coins in coin_str_list
        """
        if len(coin_types) != len(coin_values):
            raise ValueError("Coin Types List does not match size of Coin Values List")

        total_val = 0
        #Loop over coins in list
        for coin in coin_str_list:
            type_index = 0
            #Test coin over each coin type and add value of that coin type
            for coin_type in coin_types:
                if coin_type == coin:
                    total_val += coin_values[type_index]
                    continue
                type_index += 1

        return round(total_val, 2)

    ## Game loop
    running = True
    while running:

        #1 Process input/events
        clock.tick(FPS)     ## will make the loop run at the same speed all the time
        for event in pygame.event.get():        # gets all the events which have occured till now and keeps tab of them.
            ## listening for the the X button at the top
            if event.type == pygame.QUIT:
                running = False

            #Resize window
            if event.type == pygame.VIDEORESIZE:
                w_width = event.w
                w_height = event.h

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = False if paused else True

        #2 Update
        all_sprites.update()

        #3 Draw/render
        screen.fill(BLACK)

        all_sprites.draw(screen)
        ########################

        ### Your code comes here

        ########################

        #Get Image from Camera
        result, image = cam.read()
        
        #Convert to Grayscale
        gray_img = cv_image_to_gray(image)
        gray_surf = cv_image_to_surface(gray_img)

        #Get Circle Images
        if paused is False and update == 0:

            #Get Circles in image
            circles = cv_image_get_circles(gray_img, 
                                        gradient=GRADIENT, 
                                        dp=DP, 
                                        minDist=MINDIST, 
                                        param1=PARAM1, 
                                        param2=PARAM2, 
                                        minRadius=MINRADIUS, 
                                        maxRadius=MAXRADIUS
            )
            circles = cv_filter_circles(circles)

            #Filter Circles
            if circles is not None:
                circles = filter_circles(circles, True)

            #Get Circle Images
            coin_circles, coin_images_nd, coin_images_py = cv_create_circle_images(gray_img, 
                                                                                circles, 
                                                                                IMAGE_WIDTH, 
                                                                                IMAGE_HEIGHT, 
                                                                                IMAGE_BUFFER
            )

            #Create Prediction
            if coin_images_nd is not None:
                if len(coin_images_nd) > 0:
                    coin_images_nd_normal = cv_image_normalize(coin_images_nd)
                    pred = network_predict(model, coin_images_nd_normal)
                    coin_pred_str = []
                    coin_pred = []
                    for p in pred:
                        coin_pred.append(p)
                        pred_str = get_name_from_number(p, coin_types_array)
                        coin_pred_str.append(pred_str)

        #Increase update for slow updates
        update = update + 1 if update < UPDATE_RATIO else 0

        #Add Images to canvas
        screen.blit(gray_surf, (0,0))
        h_count = 0
        w_count = 1
        if coin_images_py is not None:
            for i in coin_images_py:
                if IMAGE_HEIGHT*(h_count + 1) > w_height:
                    h_count = 0
                    w_count += 2
                screen.blit(i, (w_width-IMAGE_WIDTH*w_count, IMAGE_HEIGHT*h_count))
                h_count += 1
        
        #Draw Circles on canvas
        if coin_circles is not None:
            for (x, y, r) in coin_circles:
                pygame.draw.circle(screen, GREEN, (x, y), r, 5)

        #Draw Predictions on canvas next to circle image
        h_count = 0
        w_count = 2
        for p in coin_pred_str:
            if IMAGE_HEIGHT*(h_count + 1) > w_height:
                h_count = 0
                w_count += 2
            pred_text = font.render(f"{p}", True, WHITE, BLACK)
            pred_pos = (w_width-IMAGE_WIDTH*w_count + FONT_SIZE, IMAGE_HEIGHT*h_count + IMAGE_HEIGHT/2)
            screen.blit(pred_text, pred_pos)
            h_count += 1

        #Draw Predictions on canvas on main gray image
        count = 0
        for p in coin_pred_str:
            #Get position of green circle
            x, y, r = -1000, -1000, 1
            if coin_circles is not None and count < len(coin_circles):
                x, y, r = coin_circles[count]

            #Display text at green circle
            pred_text = font.render(f"{p}", True, WHITE, BLACK)
            pred_pos = (x - r, y + r)
            screen.blit(pred_text, pred_pos)
            count += 1
        

        #Draw Total on Canvas
        total_val = calc_total_value(coin_pred_str, COIN_TYPES, COIN_VALUES)

        #Update Total Value Buffer
        if len(total_value_buffer) > TOTAL_VALUE_BUFFER_SIZE:
            total_value_buffer.pop(0)
        total_value_buffer.append(total_val)
        sorted_buffer = total_value_buffer.copy()
        sorted_buffer.sort()
        median = 0
        mean = 0
        mode = 0
        if sorted_buffer is not None:
            if len(sorted_buffer) > 0:
                median = statistics.median(sorted_buffer)
                mean = statistics.mean(sorted_buffer)
                mode = statistics.mode(sorted_buffer)

        #Calculate confidence of network. The more values at agree the more confident
        confidence_amt = 0
        if abs(mode - total_val) < TOLERANCE:
            confidence_amt += 1
        if abs(mode - median) < TOLERANCE:
            confidence_amt += 1
        if abs(mode - mean) < TOLERANCE:
            confidence_amt += 1


        #Update Test Stats
        start_text = font.render(f"Press 'SPACE' to start/stop counter.", True, WHITE, BLACK)
        mod_text = font_title.render(f"Total Value (Mode): ${mode}", True, WHITE, BLACK)

        total_text = font.render(f"Total Value (Raw): ${total_val}", True, confidence[confidence_amt], BLACK)
        med_text = font.render(f"Total Value (Median): ${median}", True, confidence[confidence_amt], BLACK)
        men_text = font.render(f"Total Value (Mean): ${round(mean, 2)}", True, confidence[confidence_amt], BLACK)
        
        start_pos = (0, 0)
        mod_pos = (0, w_height - FONT_SIZE*5)
        total_pos = (0, w_height - FONT_SIZE*3)
        med_pos = (0, w_height - FONT_SIZE*2)
        men_pos = (0, w_height - FONT_SIZE)

        screen.blit(start_text, start_pos)
        screen.blit(total_text, total_pos)
        screen.blit(med_text, med_pos)
        screen.blit(men_text, men_pos)
        screen.blit(mod_text, mod_pos)

        ## Done after drawing everything to the screen
        pygame.display.flip()       

    pygame.quit()

if __name__ == "__main__":
    main()
