from itertools import cycle
import random
import sys
import time
import datetime
import logging
import os

import pygame
from pygame.locals import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import json

from genome import Genome
#from network import Network
#from population import Population
from config import Config

import pickle
import numpy as np
import subprocess

from colorama import *
from operator import add
from operator import sub
from operator import truediv


today = "save-1"#+ str(datetime.date.today()) + "_" + time.strftime("%X")

if not os.path.exists(today):
    os.makedirs(today)

savestat = True
fpsspeed=3
FPS = 4000

bestFitness = 0
fitnessovergeneration = []
fittestovergeneration = []

#detectionOffset = 36
detectionOffset = 40

SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image and hitmask  dicts
IMAGES, HITMASKS = {}, {}
DIEIFTOUCHTOP = True
# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

asciiart=""" FLAPPY AI """
def printc(text,color):
    if color == "red":
         print(Fore.RED)
    if color == "blue":
         print(Fore.BLUE)
    if color == "green":
         print(Fore.GREEN)
    print(text)
    print(Style.RESET_ALL)

def main():

    POP_SIZE = 100
    generations = 500  # No of generations
    Model = list()
    best_individual  = Genome()
    best_fitness = -1          # Initially the best score is 0

    # There are 2 types of score to maintain
    # 1- The current_best score
    # 2- The global best score
    # Hence there will be corrensponding individuals for this score

    global_best_fitness = -1
    current_best_fitness = -1

    global_best = Genome()          # Best individual of all generations
    current_best = Genome()         # Current best individual of the lot

    c1 = 2 #np.random.randint(0,2)
    c2 = 2 #np.random.randint(0,2)

    initPygame(" AI Learning Flappy ")

    IP = list()
    for i in range(POP_SIZE):                  # Training the first set of batch
        genome = Genome()
        #genome.mutate()
        genome = playGame(genome)
        # here also the intial population has to start playing the game
        IP.append(genome)

    P = IP                                      # Setting Population to initial population

    print(" Training of initial Population is successfull ")
    #time.sleep(10)

    maximum_fitness = -1
    average_fitness = -1
    average_fitness_list  = np.array([])
    maximum_fitness_list = np.array([])
    all_generations = np.array([])

        #****************************STARTING THE GENERATION LOOP*************************************#

    for g in range(generations) :          # The iteration for each generation
        #print(" Printing generation number ",g)
        #time.sleep(10)
        metric = list()
        for m in range(POP_SIZE) :                # training each model from the population
            #print(type(m))
            #print(type(P[m]))
            P[m] = playGame(P[m])                   #instead of train , the model has to play
            metric.append([m,P[m].fitness])         # instead of test_error , append its fitness

            if  current_best_fitness  < P[m].fitness :
                current_best_fitness = P[m].fitness
                current_best_weights = P[m].get_weights()
                current_best.set_weights(current_best_weights)

            informationforscreen = {
            'generation' : g,
            'birdnumber' : m,
            'lastfitness' : P[m].fitness,
            'lastgenerationaveragefitness' : average_fitness,
            'bestfitness' : current_best_fitness
            }
            updateScreen(informationforscreen)

        values = np.array(metric)
        results =  values.reshape(values.shape[0],values.shape[1])      # setting up all the results
        average_fitness = np.sum(results[:,1],axis = 0)/100          # calculating the average fitness
        #maximum_fitness = np.amax(results[:,1],axis = 0)            # calcualting the maximum fitness

        individuals = results

        if global_best_fitness < current_best_fitness :
            global_best_fitness = current_best_fitness
            global_best_weights = current_best.get_weights()
            global_best.set_weights(global_best_weights)

        average_fitness_list = np.append(average_fitness_list,average_fitness)
        maximum_fitness_list = np.append(maximum_fitness_list,current_best_fitness)
        all_generations = np.append(all_generations,g)


        print(" The score of all people is ")
        print(individuals)

        progeny = list()

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UPDATE THE FLOCK OF BIRDS IN PSO ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
        for pop in P :
            #new_guy = Genome()
            ran1 = random.randrange(0,2)
            ran2 = random.randrange(0,2)
            init_weights = pop.get_weights()
            global_weights = global_best.get_weights()
            current_weights = current_best.get_weights()

            weight_diff_from_global = list(map(sub,global_weights,init_weights))
            weight_diff_from_current = list(map(sub,current_weights,init_weights))

            z1 = c1*ran1
            z2 = c2*ran2

            new_guy_weights_additive = list(map(add,map(lambda x:x*z1,weight_diff_from_global),map(lambda x:x*z2,weight_diff_from_current)))
            new_guy_weights = list(map(add,init_weights,new_guy_weights_additive))

            #new_guy.set_weights(new_guy_weights)

            pop.set_weights(new_guy_weights)

            #progeny.append(new_guy)


        #children = progeny

        # NO MUTATION IN PSO
        #r1 = np.random.randint(1,1000)
        #r2 = np.random.randint(0,5)


        #if(r1 < 5) :
        #    children[r2].mutate()    # call a random genome of child and mutate it

        #P = children

    metric_final = list()

    Model = P
    ############################# END OF GENERATION LOOP ##################################################
    for m in range(len(P)) :
        P[m] = playGame(P[m])                   #instead of train , the model has to play
        metric_final.append([m,P[m].fitness])         # instead of test_error , append its fitness


    values_final = np.array(metric_final)
    results_final = values_final.reshape(values_final.shape[0],values_final.shape[1])
    individuals_final = results_final


    #ranked_individuals_final = individuals_final[individuals_final[:,1].argsort()]

    #print(" Final ranked individuals ")
    #print(ranked_individuals_final)
    #print(type(ranked_individuals_final))
    #best_individual = int(ranked_individuals_final[0][0])

    #`````````````````````````````` PLOTTING THE CONVERGENCE OF ALL THE BIRDS ```````````````````````
    f_handle = open('max_pso.txt','w')

    np.save('max_pso.npy',maximum_fitness_list)

    f_handle.close()

    f_handle = open('avg_pso.txt','w')

    np.save('avg_pso.npy',average_fitness_list)

    f_handle.close()

    f_handle = open('generations.txt','w')

    np.save('gen.npy',all_generations)

    f_handle.close()

    plt.figure(2)
	#plt.subplot(211)
    plt.plot(all_generations,maximum_fitness_list,color = 'red',label = 'maximum fitness')
    plt.plot(all_generations,average_fitness_list,color = 'blue',label = 'average fitness')
    plt.xlabel("Generation number")
    red_patch = mpatches.Patch(color = 'red',label = 'minimum fitness')
    blue_patch = mpatches.Patch(color = 'blue',label = 'average fitness')
    plt.legend(handles = [red_patch,blue_patch])
    plt.ylabel("Fitness")


	#plt.subplot(212)
	#plt.plot(X,Y,color = 'green')
	#plt.xlabel("X-value")
	#plt.ylabel("Y = exp(-3x) + sin(6*pi*x)")
    plt.show()


def initPygame(screencaption):
        global SCREEN, FPSCLOCK
        pygame.init()
        init()
        FPSCLOCK = pygame.time.Clock()
        SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        pygame.display.set_caption(screencaption)

        # numbers sprites for score display
        IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # base (ground) sprite
        IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

def playGame(genome):

    """Info pour le jeux"""
    # select random background sprites
    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )
    """Info pour lancer le jeux sans le message au depart"""

    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)
    playerShmVals = {'val': 0, 'dir': 1}
    basex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    movementInfo = {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }
    #Update the network with current genes
    #genome.network.fromgenes(genome.genes)
    genome.set_weights(genome.get_weights())
    crashInfo = mainGame(movementInfo,genome)
    #fitness = showGameOverScreen(crashInfo)
    genome = crashInfo['genome']

    if Config.fitnessIsScore:
        genome.fitness = crashInfo['score']

    if genome.fitness < 0:
        genome.fitness = 0
    return genome

def mainGame(movementInfo,genome):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    framesurvived = 0

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                #Store Stat
                if savestat==True:
                    reportStat()

                pygame.quit()
                if savestat==True:
                    showStat(today)
                else:
                    sys.exit()
            if event.type == KEYDOWN and event.key == K_UP:
                if fpsspeed < 4:
                    global fpsspeed
                    fpsspeed += 1
            if event.type == KEYDOWN and event.key == K_DOWN:
                if fpsspeed != -2:
                    global fpsspeed
                    fpsspeed -= 1
        #Evaluate the NN
        if playerx < lowerPipes[0]['x'] + detectionOffset:
            nextPipe = lowerPipes[0]
        else:
            nextPipe = lowerPipes[1]

        nextPipeY = float(SCREENHEIGHT - nextPipe['y']) / SCREENHEIGHT

        playerYcorrectAxis = float(SCREENHEIGHT - playery) / SCREENHEIGHT
        distanceBetweenPlayerAndNextPipe = float(nextPipe['x'] - playerx)/ SCREENWIDTH

        NNinput = np.array([[playerYcorrectAxis],[nextPipeY]])

        NNoutput = genome.predict(NNinput)
        #print("Printing output ")
        #print(NNoutput)

        if NNoutput > 0.5:
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True
                #print(" I flapped ")

        info = {'playery': playerYcorrectAxis, 'pipey': nextPipeY, 'distance': distanceBetweenPlayerAndNextPipe}


        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        if crashTest[0] or playery < 5:
            genome.fitness = framesurvived
            #print(" My fitness is ",framesurvived)
            #time.sleep(3)
            return {
                'score': score,
                'genome': genome
            }

        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        playerHeight = IMAGES['player'][playerIndex].get_height()
        if playery > 5:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        if Config.debug:
            displayInfo(info)

        framesurvived += 1
        #print(" Hey I survived frames = ",framesurvived)
        #time.sleep(10)
        SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))


        global FPS
        if fpsspeed==4:
            #No FPS clock ticking, may be instable
            continue
        if fpsspeed==3:
            FPS=4000
        if fpsspeed==2:
            FPS=400
        if fpsspeed==1:
            FPS=40
        if fpsspeed==0:
            FPS=30
        if fpsspeed==-1:
            FPS=15
        if fpsspeed==-2:
            FPS=3

        pygame.display.update()
        FPSCLOCK.tick(FPS)

def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1

def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]

def displayInfo(info):
    ###Display useful info : the input for the ANN
    myfont = pygame.font.Font(None, 30)
    # render text
    playery = str(info['playery'])
    tubey = str(info['pipey'])
    distance = str(info['distance'])

    labelplayery = myfont.render(playery,1,(255,255,0))
    labeltubey = myfont.render(tubey,1,(0,255,255))
    labeldistance = myfont.render(distance,1,(255,255,255))

    SCREEN.blit(labelplayery, (SCREENWIDTH / 2 - 100, SCREENHEIGHT * 0.7))
    SCREEN.blit(labeltubey, (SCREENWIDTH / 2  - 100, SCREENHEIGHT * 0.8))
    SCREEN.blit(labeldistance, (SCREENWIDTH / 2 - 100, SCREENHEIGHT * 0.9))

def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def reportStat():
    with open(today + '/fitnessovergeneration.dat', 'wb') as handle:
        pickle.dump(fitnessovergeneration, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(today + '/fittestovergeneration.dat', 'wb') as handle:
        pickle.dump(fittestovergeneration, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(today + '/bestfitness.dat', 'wb') as handle:
        pickle.dump(bestFitness, handle, protocol=pickle.HIGHEST_PROTOCOL)

def updateScreen(info):
    #Clear the screen
    #subprocess.call(["printf", "\033c"])
    #subprocess.call(["printf", "cls"])
    os.system('cls')
    #Print asciiart
    printc(asciiart,"green")

    if info["generation"] > 1:
        print("----Last generation----")
        printc("Average fitness: %s" % str(info["lastgenerationaveragefitness"]), "blue")
        print("-----------------------")
    if info["birdnumber"] > 1:
        printc("Last Fitness: %s" % str(info["lastfitness"]), "green")

    printc("Best Fitness: %s" % str(info["bestfitness"]),"red")
    print("----Status----")
    printc("Generation number : %s/%s" % (str(info["generation"]),str(Config.maxGeneration)),"blue")
    printc("Bird number: %s/%s" % (str(info["birdnumber"]), str(Config.numberOfIndividuals)),"blue")

def showStat(folder):

    fitnessovergeneration = pickle.load(open(folder + '/fitnessovergeneration.dat', 'rb'))
    fittestovergeneration = pickle.load(open(folder + '/fittestovergeneration.dat', 'rb'))
    bestfitness = pickle.load(open(folder + '/bestfitness.dat', 'rb'))

    #Clear the screen
    subprocess.call(["printf", "\033c"])
    printc("Statistics of %s" % folder,"blue")
    printc("-" * 20,"green")
    print ("Number of generation: %s" % len(fittestovergeneration))
    printc("Best Fitness: %s" % bestFitness,"red")

    plt.figure(1)
    plt.plot(fittestovergeneration)
    plt.plot(fitnessovergeneration)
    plt.show()
    sys.exit()




if __name__ == '__main__':
    main()
