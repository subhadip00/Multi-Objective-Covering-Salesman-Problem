from __future__ import division
import kmedoids
import math
import random
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
import copy
import numpy as np 

mutation_probability=0.1
# We are taking 2 functions which are conflicting. One is coverage and another is tour-length
# We shall minimise the tour length and maximise the coverage
# Tour length is plotted in y axis and coverage is plotted in x axis.
def distance(point_1,point_2):
    return np.sqrt((abs(point_1[0]-point_2[0]))**2 + (abs(point_1[1]-point_2[1]))**2)

#Calculation of tour length is the first objective
def tour_length(route):
    length=0
    for i in range(len(route)-1):
        length=length+ distance(route[i],route[i+1])
    length=length+ distance(route[0],route[len(route)-1])
    return length

# to check if a point is within a certain radius is an auxilary function for calculating the coverage
def check_within_radius(temp):
    #temp is medoid point
    #x is any random point
    
    points=[]
    for x in data:
        if (((temp[0]-x[0])**2) + ((temp[1]-x[1])**2))< radius*radius:
            points.append(x)
    
    return points


# Calculation of coverage is the 2nd objective
def calculate_coverage(medoid):
    covered_point=[]
    for temp in medoid:
        x=check_within_radius(temp)
        for item in x:
            covered_point.append(item)
    temp=Counter([tuple(x) for x in covered_point])
    z=[list(k) for k, v in temp.items() if v >= 1]
    c=len(z)/float(len(data))
    return (c*100)
        

def fast_non_dominated_sort(coverage, tl):
    S=[[] for i in range(0,len(coverage))]
    front = [[]]
    n=[0 for i in range(0,len(coverage))]
    rank = [0 for i in range(0, len(coverage))]

    for p in range(0,len(coverage)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(coverage)):
            if (coverage[p] > coverage[q] and tl[p] < tl[q]) or (coverage[p] >= coverage[q] and tl[p] < tl[q]) or (coverage[p] > coverage[q] and tl[p] <= tl[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (coverage[q] > coverage[p] and tl[q] < tl[p]) or (coverage[q] >= coverage[p] and tl[q] < tl[p]) or (coverage[q] > coverage[p] and tl[q] <= tl[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:# Taking each element of the front
            for q in S[p]: # We are checking the points which that point dominates
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1] # deleting the last front which is empty
    return front


def index_of(a,list1): # returns the index of a particular element
    for i in range(0,len(list1)):
        if list1[i] == a:
            return i
    

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1: # Find at which index minimum value is present in list1
            sorted_list.append(index_of(min(values),values)) # append index of minimum values in sorted_list
        values[index_of(min(values),values)] = math.inf #The minimum value is set to infiite so that it can't be minimum value any more
    return sorted_list # Return the index of sorted list

def crowding_distance(values1, tl, front):
# A front has a few number of populations. Each population has a crowding distance. Initially, set crowding distnace of each population to be 0
	distance = [0 for i in range(0,len(front))] 
	sorted1 = sort_by_values(front, values1[:])
	sorted2 = sort_by_values(front, tl[:])
	distance[0] = 4444444444444444
	distance[len(front) - 1] = 4444444444444444
	for k in range(1,len(front)-1):
	    #max(values1) is maximum value of objective 1. sorted1 is used to take value according to objective 1
	    distance[k] = distance[k]+ (values1[sorted1[k+1]] - tl[sorted1[k-1]])/(max(values1)-min(values1)+1)
	for k in range(1,len(front)-1):
	    distance[k] = distance[k]+ (values1[sorted2[k+1]] - tl[sorted2[k-1]])/(max(tl)-min(tl)+1)
	return distance


def crossover(parent1,parent2):
	child1=[]
	child2=[]
	min_length=min(len(parent1),len(parent2))
	crossover_point=np.random.randint(0,min_length)
	for i in range(crossover_point+1):
		if parent1[i] not in child1:
			child1.append(parent1[i])
		if parent2[i] not in child2:
			child2.append(parent2[i])
	
	for i in range(crossover_point+1,len(parent2)):
		if parent2[i] not in child1:
			child1.append(parent2[i])

	for i in range(crossover_point+1,len(parent1)):
		if parent1[i] not in child2:
			child2.append(parent1[i])

	if len(child1)<k_min:
		while len(child1)!=k_min:
			x=np.random.randint(0,len(data))
			if data[x] not in child1:
				child1.append(data[x])

	if len(child2)<k_min:
		while len(child2)!=k_min:
			x=np.random.randint(0,len(data))
			if data[x] not in child2:
				child2.append(data[x])
	return child1,child2
	
def mutated_gene(gene,chromosome):
    #print("Mutation done")
    index_of_gene=index_of(gene,data)
    distance_of_other_points_from_that_gene=D[index_of_gene]
    sorted_index=[]
    for item1 in sorted(distance_of_other_points_from_that_gene):
        sorted_index.append(index_of(item1,distance_of_other_points_from_that_gene))

    no_of_neighbour=0
    while(no_of_neighbour<3):
        i=0
        while True:
            if data[sorted_index[i]] not in chromosome:
                coor_x=data[sorted_index[i]]
                no_of_neighbour+=1
                i=i+1
                break
            else:
                i=i+1
        while True:
            if data[sorted_index[i]] not in chromosome:
                coor_y=data[sorted_index[i]]
                no_of_neighbour+=1
                i=i+1
                break
            else:
                i=i+1
        while True:
            if data[sorted_index[i]] not in chromosome:
                coor_z=data[sorted_index[i]]
                no_of_neighbour+=1
                i=i+1
                break
            else:
                i=i+1
    
    taken_gene=random.choice([coor_x,coor_y,coor_z])

    return taken_gene


def three_nearest_neighbour_mutation(chromosome):
    temp=copy.deepcopy(chromosome)
    for i in range(len(temp)):
        gene=temp[i]
        random_no=np.random.random()
        if random_no<mutation_probability:
            gene1=mutated_gene(gene,temp)
            temp[i]=gene1
        else:
            temp[i]=gene

    return temp 
'''
def mutate_population(population):
    mutated_population=[]
    for i in range(len(population)):
        a=np.random.random()
        if a<mutation_probability:

            min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse=utility_min_max(population)

            temp=three_nearest_neighbour_mutation(population[i],min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse)
            mutated_population.append(temp)
        else:
            mutated_population.append(population[i])
        
    return mutated_population
'''        
	 


# Actual  program starts here

with open('/content/drive/My Drive/Colab Notebooks/SJC818.txt') as f:
	data=[]
	for line in f:
		x,y=(line.strip('\n').split())
		data.append([int(x),int(y)])


population_size=50

print("NSGA II based process")
k_min=165 #Minimum number of cluster
k_max=198 #maximum number of cluster
radius=500

print("Radius ",radius,"k_min ",k_min,"k_max ",k_max)
max_gen = 150

#Initialization

#Initial population is set. . Each individual in the population is determined randomly
D=pairwise_distances(data,metric='euclidean')
#Now we shall create initial population consisting of a certain number of individuals
population=[]
for i in range(population_size):
	no_of_cluster=np.random.randint(k_min,k_max+1)
	M, C = kmedoids.kMedoids(D,no_of_cluster)
	medoid=[]
	for item in M:
		medoid.append(data[item])
	if medoid not in population:
		population.append(medoid)

gen_no=0

#while loop runs till maximum generation
while(gen_no<max_gen):
    coverage = [calculate_coverage(population[i]) for i in range(0,population_size)]
    tl = [tour_length(population[i]) for i in range(0,population_size)]
    non_dominated_sorted_population = fast_non_dominated_sort(coverage[:],tl[:])
    print(gen_no+1)
   
    crowding_distance_values=[]
    #non_dominated_sorted_population[i][:]) ---> It indicates one particular front
    for i in range(0,len(non_dominated_sorted_population)):
        crowding_distance_values.append(crowding_distance(coverage[:],tl[:],non_dominated_sorted_population[i][:]))
    population2 = copy.deepcopy(population[:]) #Initially all the population of population is appended to population2
    #Generating offsprings
    # Now new individuals are added to make the population size twice
    for i in range(population_size):
        a1 = random.randint(0,population_size-1)
        b1 = random.randint(0,population_size-1)
        while(a1==b1):
        	b1 = random.randint(0,population_size-1)
        child1,child2=crossover(population[a1],population[b1])

        flag_for_child1=0
        flag_for_child2=0

        if np.random.random()<mutation_probability:
            mutated_child1=three_nearest_neighbour_mutation(child1)
            population2.append(mutated_child1)
            flag_for_child1=1

        if flag_for_child1==0:
            population2.append(child1)

        if np.random.random()<mutation_probability:
            mutated_child2=three_nearest_neighbour_mutation(child2)
            population2.append(mutated_child2)
            flag_for_child2=1

        if flag_for_child2==0:
            population2.append(child2)

        
        

    # Now length of coverage2 will be double as function_vales
    coverage2 = [calculate_coverage(population2[i])for i in range(0,2*population_size)]
    tl2 = [tour_length(population2[i])for i in range(0,2*population_size)]

    # Again non-dominated is sorting is done on twice of population size
    non_dominated_sorted_population2 = fast_non_dominated_sort(coverage2[:],tl2[:])

    #crowding distances is calculated
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_population2)):
        crowding_distance_values2.append(crowding_distance(coverage2[:],tl2[:],non_dominated_sorted_population2[i][:]))


    # Now we shall select the number of individual which is same as initial population from twice of population size
    new_population= []
    for i in range(0,len(non_dominated_sorted_population2)):
        # each front of non_dominated_sorted_population2 will be traversed
        non_dominated_sorted_population2_1 = [index_of(non_dominated_sorted_population2[i][j],non_dominated_sorted_population2[i] ) for j in range(0,len(non_dominated_sorted_population2[i]))]
        #front22 will contain indices 
        front22 = sort_by_values(non_dominated_sorted_population2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_population2[i][front22[j]] for j in range(0,len(non_dominated_sorted_population2[i]))]
        front.reverse()
        
        for value in front:
            new_population.append(value)
            if(len(new_population)==population_size):
                break
        if (len(new_population) == population_size):
            break
    population = [population2[i] for i in new_population]


    gen_no = gen_no + 1



c=[coverage[item] for item in non_dominated_sorted_population[0] ]
t=[tl[item] for item in non_dominated_sorted_population[0] ]


euclidean_distance=[]


for i in range(len(c)):
    sum1=0
    for j in range(len(c)):
        p1=[c[i],t[i]]
        p2=[c[j],t[j]]
        sum1+=distance(p1,p2)
    #print("sum1= ",sum1)
    euclidean_distance.append(sum1)

min1=min(euclidean_distance)

min_index=index_of(min1,euclidean_distance)

solution=[t[min_index],c[min_index]]
print("solution ",solution)

print("coverage",c)
print("tour length",t)
plt.plot(c)
plt.xlabel("Coverage plot")
plt.show()

plt.plot(t)
plt.xlabel("Tour length plot")
plt.show()

plt.xlabel('coverage', fontsize=15)
plt.ylabel('Tour Length', fontsize=15)
plt.scatter(c, t)
plt.show()
#drive.flush_and_unmount
