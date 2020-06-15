#!/usr/bin/env python3

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from collections import Counter

####################### DATATYPE DEFINITIONS #######################
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    def __lt__(self, other):
        if self.x < other.x:
            return True
        elif self.x == other.x and self.y < other.y:
            return True
        else:
            return False


#Used to evaluate the fitness of a route
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0

    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())

        return self.fitness

#Used in the child generation phase
childMethods = {0:"PMX", 1:"EX", 2:"OX"}
mutationMethods = {0:"No", 1:"Swap", 2:"Inversion"}

####################### FUNCTION DEFINITIONS #######################
def createRoute(cityList):
    '''
    Narrative: Generates a random route from all of the cities and returns it as a list.

    Parameters:
        cityList: A list of all the cities.
    '''
    route = random.sample(cityList, len(cityList))

    return route


def initialPopulation(popSize, cityList):
    '''
    Narrative: Generates a population of potential routes and returns it in a list.

    Parameters:
        popSize: An integer that determines how many routes to create.
        cityList: A list of all the cities.
    '''
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    '''
    Narrative: Calculates the fitness of all the routes in the population and returns them in a sorted
                list with the fittest at position 0.

    Parameters:
        population: A list of all the routes in the current generation.
    '''
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):
    '''
    Narrative: Determines .

    Parameters:
        popRanked: A list of all the cities
        eliteSize: An integer for the number of fittest routes to carry over in simulation
    '''
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break

    return selectionResults


def matingPool(population, selectionResults):
    '''
    Narrative: Determines .

    Parameters:
        population: A list of all the routes in the current generation.
        selectionResults:
    '''
    matingpool = []

    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])

    return matingpool


def breedPopulation(matingpool, eliteSize, childMethod):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        if childMethod == 0:
            child = PMX(pool[i], pool[len(matingpool) - i - 1])
        elif childMethod == 1:
            child = EX(pool[i], pool[len(matingpool) - i - 1])
        elif childMethod == 2:
            child = OX(pool[i], pool[len(matingpool) - i - 1])

        children.append(child)

    return children


def mutatePopulation(population, mutationRate, mutMethod):
    '''
    Narrative: Has the possibility of triggering a change in the routes within population and
                returns the new population.

    Parameters:
        population: A list of all the routes in the current generation.
        mutationRate: A value between 0.0-1.0.  Determines the likelihood of a mutation
                        occuring.
        mutMethod: An integer that determines what mutation method to use.
    '''
    mutatedPop = []

    for ind in range(0, len(population)):
        if mutMethod == 0:
            mutatedInd = population[ind]
        elif mutMethod == 1:
            mutatedInd = SwapMutation(population[ind], mutationRate)
        elif mutMethod == 2:
            mutatedInd = InversionMutation(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)

    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate, childMethod, mutMethod):
    '''
    Narrative: Produces the next generation in the simulation and returns it.

    Parameters:
        currentGen: A list of all the routes in the current generation.
        eliteSize: An integer that determines how many of the current fittest routes carry
                    over into the next generation.
        mutationRate: A value between 0.0-1.0.  Determines the likelihood of a mutation
                        occuring.
        childMethod: An integer that determines what recombintion method to use.
        mutMethod: An integer that determines what mutation method to use.
    '''
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize, childMethod)
    nextGeneration = mutatePopulation(children, mutationRate, mutMethod)

    return nextGeneration


####################### RECOMBINATION FUNCTIONS #######################
#Partially Mapped Crossover method of recombination/child generation
def PMX(parent1, parent2):
    '''
    Narrative: Copies a section of parent1's route into child, then looks at parent2 in the
                same section and places the unassigned values in that region of child in the
                location of the value that took its place.
                    ex.) if 4 was copied from parent1 into child and in parent2 the spot
                            4 was copied into was where 8 goes in parent2, then the position
                            of 4 in parent2 would be found and the 8 would be placed in that
                            location in child
                Once those values are placed, the remaining values in parent2 that have not
                been copied into child are copied over in the same order they are in in parent2.
                Returns the new list.

    Parameters:
        parent1 & 2: Lists of cities to be used to generate a new route, child.
    '''
    child = [None] * len(parent1)

    #Choose 2 crossover points at random
    geneA, geneB = random.choices(range(len(parent1)),k=2)

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    tmpIndex = 0
    i = None #Holds the gene in parent2 that needs placed
    j = None #Holds the gene in parent1 that is in the place i was in parent 2
    k = None #Used to find a position to place i in the child

    child[startGene:endGene] = parent1[startGene:endGene]

    for index in range(startGene, endGene):
        if parent2[index] not in parent1[startGene:endGene]:
            i = parent2[index]
            j = parent1[index]
            tmpIndex = parent2.index(j)

            while startGene <= tmpIndex and tmpIndex < endGene:
                k = parent1[tmpIndex]
                tmpIndex = parent2.index(k)

            child[tmpIndex] = i

    for index in range(0, len(parent2)):
        if child[index] == None:
            child[index] = parent2[index]

    return child

def CalcIndex(lst, cIndex):

    lIndex = cIndex - 1
    rIndex = cIndex + 1

    if lIndex < 0:
        lIndex = len(lst) - 1

    if rIndex >= len(lst):
        rIndex = 0

    dictry = {"left":lIndex, "right":rIndex}

    return dictry


def GenerateEdgeTable(parent1, parent2):
    edges = {x: [] for x in parent1}

    cIndex = None
    sideIn = None

    for index in range(0, len(parent1)):
        sideIn = CalcIndex(parent1, index)

        edges[parent1[index]].append(parent1[sideIn["left"]])
        edges[parent1[index]].append(parent1[sideIn["right"]])

        cIndex = parent2.index(parent1[index])
        sideIn = CalcIndex(parent2, cIndex)

        edges[parent1[index]].append(parent2[sideIn["left"]])
        edges[parent1[index]].append(parent2[sideIn["right"]])

    return edges


def FindShortestListKey(lst, edges, child):
    #NOTE: Given that there can be multiple instances of an element in the lists to signify a common edge,
    #          need to remove the duplicate edges when checking their length
    shortest = []
    size = None
    item = None
    rmDups = None

    for key in lst:
        if key not in child:
            rmDups = Counter(edges[key])

            if size == None:
                size = len(rmDups)
                shortest.append(key)

            elif len(rmDups) > 0 and len(rmDups) < size:
                size = len(rmDups)
                shortest = [key]

            elif len(rmDups) == size:
                shortest.append(key)

    if len(shortest) > 1:
        item = random.choice(shortest)

    else:
        item = shortest[0]

    return item


#Edge Crossover method of recombination/child generation
def EX(parent1, parent2):
    '''
    Narrative: Generates a new route, child, from parent1 and 2 by prioritizing shared
                edges (ex. a road between city A and B in both parent1 and 2) and, when
                one isn't found, picking a city that connects to the smallest number of
                cities.  When there is a tie, or no city that matches the previous critieria,
                a city is randomly chosen from the available options.  Once a city is selected,
                it is removed from the list.  This process is repeated until there are no cities
                left to add.  Returns the new list.

    Parameters:
        parent1 & 2: Lists of cities to be used to generate a new route, child.
    '''
    child = []

    #A dictionary with each city as a key that connects to a list of its edges
    edges = None

    #Used to create child
    entry = None
    currElement = None
    count = None
    i = 0
    unused = []

    #1) Construct Edge Table
    edges = GenerateEdgeTable(parent1, parent2)

    #2) Pick a initial element @ random & put in offspring
    entry = random.choice(list(edges))
    child.append(entry)


    #3) Set the variable current_element = entry
    currElement = entry

    while len(child) < len(parent1):

        #4) Remove all references to current_element from table
        for key in edges:
            try:
                while True:
                    edges[key].remove(currElement)

            except ValueError:
                pass

        #5) Examine list for current_element
        #NOTE: Ties are split at random
        if len(edges[currElement]) != 0:
            count = Counter(edges[currElement]).most_common()

            #5.1) If there is a common edge, pick that to be the next element
            if count[0][1] == 2:
                i = 1

                while i < len(count) and count[i][1] == 2:
                    i += 1

                entry = random.choice(count[0:i])[0]

                while entry in child:
                    entry = random.choice(count[0:i])[0]

                currElement = entry

            #5.2) Otherwise, pick the entry in the list which itself has the shortest list
            else:
                currElement = FindShortestListKey(edges[currElement], edges, child)

        #6) In the case of reaching an empty list, a new element is chosen at random
        else:
            unused = [item for item in parent1 if item not in child]

            currElement = random.choice(unused)

        #Reset Variables
        i = 0
        unused = []

        child.append(currElement)

    return child


#Order Crossover method of recombination/child generation
def OX(parent1, parent2):
    '''
    Narrative: Copies a section of parent1's route into child, then starting at the end of
                the section, the values of parent2 are copied over into child if they are
                not already in child while still preverving their order.  Returns the new list.

    Parameters:
        parent1 & 2: Lists of cities to be used to generate a new route, child.
    '''
    child = [None] * len(parent1)
    notAdded = None

    #Choose 2 crossover points at random
    geneA, geneB = random.choices(range(len(parent1)),k=2)

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    p2Index = endGene
    cpIndex = endGene

    #Copy over segment from parent1 into child
    child[startGene:endGene] = parent1[startGene:endGene]

    #Start from the end of the crossover point in parent 2 and copy over any elements not currently in child
    notAdded = [item for item in parent2 if item not in child]

    for i in range(0, len(notAdded)):
        if parent2[p2Index] not in child:
            if child[cpIndex] != None:
                while child[cpIndex] != None:
                    cpIndex += 1

                    if cpIndex >= len(child):
                        cpIndex = 0

        else:
            while parent2[p2Index] in child:
                p2Index += 1

                if p2Index >= len(parent2):
                    p2Index = 0

        child[cpIndex] = parent2[p2Index]


        p2Index += 1
        cpIndex += 1

        if p2Index >= len(parent2):
            p2Index = 0

        if cpIndex >= len(child):
            cpIndex = 0

    return child

####################### MUTATION FUNCTIONS #######################
def SwapMutation(individual, mutationRate):
    '''
    Narrative: If a mutation is triggered, randomly selects two cities in individual and swaps
                their positions.  Returns the possibly modified list.

    Parameters:
        individual: A list of cities.  Possibly has the cities reordered.
        mutationRate: A value between 0.0-1.0.  Determines the likelihood of a mutation
                        occuring.
    '''
    newIndividual = individual.copy()

    if(random.random() < mutationRate):
        index1, index2 = random.choices(range(len(newIndividual)),k=2)

        newIndividual[index1] = individual[index2]
        newIndividual[index2] = individual[index1]
    return newIndividual


def InversionMutation2(individual, mutationRate):
    '''
    Narrative: If a mutation is triggered, randomly selects two positions and reverses the order
                of the cities between them.  Returns the possibly modified list.

    Parameters:
        individual: A list of cities.  Possibly has the cities reordered.
        mutationRate: A value between 0.0-1.0.  Determines the likelihood of a mutation
                        occuring.
    '''
    newIndividual = individual.copy()

    if(random.random() < mutationRate):
        index1, index2 = random.choices(range(len(newIndividual)),k=2)

        startIndex = min(index1, index2)
        endIndex = max(index1, index2)+1

        newIndividual[startIndex:endIndex] = reversed(individual[startIndex:endIndex])
    return newIndividual
