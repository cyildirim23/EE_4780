import pygame
import cv2 as cv
import numpy as np
import shutil
import os

from neural_network import *

#Image Constants
WIDTH = 1080
HEIGHT = 720
FPS = 30
UPDATE_RATIO = 16
CAM_PORT = 0
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_BUFFER = 4

#Hough Circle Constants
GRADIENT= cv.HOUGH_GRADIENT
DP = 1.05
MINDIST = 70
PARAM1 = 50
PARAM2 = 45
MINRADIUS = 10
MAXRADIUS = 200

#Train Constants
TRAIN_DIR = "./neural_network/train_data"
TEST_DIR = "./neural_network/test_data"
COIN_TYPES = ['penny', 'nickel', 'dime', 'quarter', 'half_dollar', 'dollar_coin']
COIN_VALUES= [   0.01,     0.05,   0.10,      0.25,          0.50,          1.00]

# Define Colors 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

#Other Constants
FONT_SIZE = 18

## initialize pygame and create window
pygame.init()
pygame.mixer.init()  ## For sound
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Coin Counter Dataset Creator")
clock = pygame.time.Clock()     ## For syncing the FPS

## group all the sprites together for ease of update
all_sprites = pygame.sprite.Group()

#Define Camera
cam = cv.VideoCapture(CAM_PORT, cv.CAP_DSHOW)
cam.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

#Define Font
font = pygame.font.Font('freesansbold.ttf', FONT_SIZE)
title_font = pygame.font.Font('freesansbold.ttf', FONT_SIZE * 2)

#Define Global Variables
paused = False
save_pictures = False
update = 0
stage = 0
c_count = 0
coin_circles = None
coin_images_nd = None
coin_images_py = None
dir = None
file_name = None
train_dir_bool = True
w_width = WIDTH
w_height = HEIGHT

def get_coin(i: int, coin_types: list, train_bool: bool):
    if i < 0:
        #If training is False (in test data) show previous as the end of training data
        if train_bool is False:
            return coin_types[len(coin_types)-1]
        else:
            return "NULL"
    elif i >= len(coin_types):
        #If in training data, show next is beginning of test data.        
        if train_bool is True:
            return coin_types[0]
        else:
            return "NULL"
    #Else return the current index of coin_types
    else:
        return coin_types[i]

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

        #Keyboard Inputs
        if event.type == pygame.KEYDOWN:

            #Press SPACE to go to the next coin
            if event.key == pygame.K_SPACE:
                #If next coin is out of range, see if need to go to test data set
                if stage >= len(COIN_TYPES) - 1:

                    #Go to test data set
                    if train_dir_bool:
                        stage = 0
                        train_dir_bool = False

                    #Else finish program
                    else:
                        running = False
                
                #Else go to next coin
                else:
                    stage += 1
            
            #Press BACKSPACE to go to previous coin
            if event.key == pygame.K_BACKSPACE:
                #If current stage is 0, see if need to go to train data set
                if stage == 0:

                    #If stage is 0 and train_dir_bool is true, then do nothing
                    if train_dir_bool:
                        stage = 0

                    #If stage is 0 and train_dir_bool is False, go to training set
                    else:
                        train_dir_bool = True
                        stage = len(COIN_TYPES) - 1

                #Else just reduce stage by 1
                else:
                    stage -= 1

            #Press ENTER to start/stop saving picures
            if event.key == pygame.K_RETURN:
                if save_pictures:
                    save_pictures = False
                else:
                    save_pictures = True

            #Press R to reset dataset for current directory
            if event.key == pygame.K_r:
                if os.path.isdir(dir):
                    print("Removing: " + dir)
                    shutil.rmtree(dir)
                os.mkdir(dir)

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
    gray_surf_size = gray_surf.get_size()

    #Calculate dir
    dir = TRAIN_DIR if train_dir_bool else TEST_DIR
    if stage < len(COIN_TYPES) and stage >= 0:
        dir = dir + "/" + COIN_TYPES[stage]
    dir_text = font.render(f"CURRENT FOLDER: {dir}", True, WHITE, BLACK)
    dir_pos = (0, gray_surf_size[1])
    
    #Calc Number of images in dir
    c_count = 0
    if os.path.isdir(dir):
        c_count = len(os.listdir(dir))
    count_text = font.render(f"CURRENT FOLDER PICTURE COUNT:    {c_count} Images", True, WHITE, BLACK)
    count_pos = (0, gray_surf_size[1] + FONT_SIZE)

    #Get file name
    if stage >= 0 and stage < len(COIN_TYPES):
        file_name = COIN_TYPES[stage]

    #Instructions Text
    inst_text = title_font.render("Instructions", True, WHITE, BLACK)
    curr_text = font.render(f"Current Coin: {get_coin(stage, COIN_TYPES, train_dir_bool)}.", True, WHITE, BLACK)
    forw_text = font.render(f"Press 'SPACE' to go to {get_coin(stage + 1, COIN_TYPES, train_dir_bool)}.", True, WHITE, BLACK)
    back_text = font.render(f"Press 'BACKSPACE' to go to {get_coin(stage - 1, COIN_TYPES, train_dir_bool)}.", True, WHITE, BLACK)
    res_text  = font.render(f"Press 'R' to reset {dir}.", True, WHITE, BLACK)
    capt_text = font.render(f"Press 'ENTER' to {'stop' if save_pictures else 'start'} saving pictures.", True, WHITE, BLACK)

    #Instruction Text Locations
    inst_pos = (0, gray_surf_size[1] + FONT_SIZE*3)
    curr_pos = (0, gray_surf_size[1] + FONT_SIZE*5)
    forw_pos = (0, gray_surf_size[1] + FONT_SIZE*6)
    back_pos = (0, gray_surf_size[1] + FONT_SIZE*7)
    res_pos  = (0, gray_surf_size[1] + FONT_SIZE*8)
    capt_pos = (0, gray_surf_size[1] + FONT_SIZE*9)
    
    #Get Circle Images
    if paused is False and update == 0:

        #Get Circles
        circles = cv_image_get_circles(gray_img, 
                                    gradient=GRADIENT, 
                                    dp=DP, 
                                    minDist=MINDIST, 
                                    param1=PARAM1, 
                                    param2=PARAM2, 
                                    minRadius=MINRADIUS, 
                                    maxRadius=MAXRADIUS
        )

        #Get Circle Images
        coin_circles, coin_images_nd, coin_images_py = cv_create_circle_images(gray_img, 
                                                                               circles, 
                                                                               IMAGE_WIDTH, 
                                                                               IMAGE_HEIGHT, 
                                                                               IMAGE_BUFFER
        )

        #Save Pictures
        if save_pictures:
            for i in coin_images_nd:
                if os.path.isdir(dir) is False:
                    os.mkdir(dir)
                cv_save_image(i, dir, file_name)

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
                w_count += 1
            screen.blit(i, (w_width-IMAGE_WIDTH*w_count, IMAGE_HEIGHT*h_count))
            h_count += 1
    
    #Draw Circles on canvas
    if coin_circles is not None:
        for (x, y, r) in coin_circles:
            pygame.draw.circle(screen, GREEN if save_pictures else RED, (x, y), r, 5)

    #Add text to canvas
    screen.blit(dir_text, dir_pos)
    screen.blit(count_text, count_pos)
    screen.blit(inst_text, inst_pos)
    screen.blit(curr_text, curr_pos)
    screen.blit(forw_text, forw_pos)
    screen.blit(back_text, back_pos)
    screen.blit(res_text, res_pos)
    screen.blit(capt_text, capt_pos)

    ## Done after drawing everything to the screen
    pygame.display.flip()       

pygame.quit()


