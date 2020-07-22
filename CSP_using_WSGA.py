import matplotlib.pyplot as plt 
from collections import Counter
import pandas as pd 
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np 
from sklearn.cluster import KMeans
import operator
import random
import copy
def index_of(item,list1):
	for i in range(len(list1)):
		if list1[i]==item:
			return i
def check_within_radius(temp):
    #temp is medoid point
    #x is any random point
    points=[]
    for x in data:
        if (((temp[0]-x[0])**2) + ((temp[1]-x[1])**2))< radius*radius:
            points.append(x)
    return points
def calculate_coverage(medoid):
    covered_point=[]
    for temp in medoid:
        x=check_within_radius(temp)
        for item in x:
            covered_point.append(item)
    temp=Counter([tuple(x) for x in covered_point])
    z=[list(k) for k, v in temp.items() if v >= 1]
    return ((len(z)/len(data))*100)
def distance(point_1,point_2):
    return np.sqrt((abs(point_1[0]-point_2[0]))**2 + (abs(point_1[1]-point_2[1]))**2)

def tour_length_and_coverage(route,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse):
    length=tour_length_only(route)
    coverage_inverse=coverage_inverse_only(route)
    length_normalised=(length-min_tourLength)/(max_tourLength-min_tourLength+1)
    #print("length normalised ",length_normalised)
    coverage_inverse_normalised=(coverage_inverse-min_coverageInverse)/(max_coverageInverse-min_coverageInverse+1)
    #print("Coverage Inverse normalised ",coverage_inverse_normalised)
    return w1*length_normalised + w2*coverage_inverse_normalised
    
def tour_length_only(route):
    length=0
    for i in range(len(route)-1):
        length=length+ distance(route[i],route[i+1])
    length=length+ distance(route[0],route[len(route)-1])
    return length
    
def coverage_inverse_only(route):
    x=calculate_coverage(route)
    return 1/x

def utility_min_max(population):
    tl=[]
    coverage_inverse=[]
    for i in range(len(population)):
        tl.append(tour_length_only(population[i]))
        coverage_inverse.append(coverage_inverse_only(population[i]))
    return min(tl),max(tl),min(coverage_inverse),max(coverage_inverse)
        
        
def rank_routes(population):
    min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse=utility_min_max(population)
    fitness_result={}
    for i in range(len(population)):
        fitness_result[i]=tour_length_and_coverage(population[i],min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse)
    return sorted(fitness_result.items(),key=lambda x:x[1])

def selection(population_fitness_dict):
    selection_result=[]
    '''
    for i in range(elite_size):
        selection_result.append(population_fitness_dict[i][0])
    '''
    mask=list(range(len(population_fitness_dict)))
    np.random.shuffle(mask)
    for i in range(len(population_fitness_dict)):
        mask=list(range(len(population_fitness_dict)))
        np.random.shuffle(mask)
        sample=random.sample(mask,2)
        if population_fitness_dict[sample[0]]<population_fitness_dict[sample[1]]:
            selection_result.append(sample[0])
        else:
            selection_result.append(sample[1])
    return selection_result


def crossover(parent1,parent2):
    child1=[]
    child2=[]
    #child1 will be same in size with parent1
    #child2 will be same in size with parent2
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
            #print("INSIDE CROSSOVER WHILE (1)")
            x=np.random.randint(0,len(data))
            if data[x] not in child1:
                child1.append(data[x])
    if len(child2)<k_min:
        while len(child2)!=k_min:
            #print("INSIDE CROSSOVER WHILE (1)")
            x=np.random.randint(0,len(data))
            if data[x] not in child2:
                child2.append(data[x])
                
    return child1,child2
            

def breed_population(selected_individual):
    children=[]
    #print("Selected individual ",selected_individual)
    #print("Length of selected individual ",len(selected_individual))
    
    for i in range((len(selected_individual))//2):
        mask=list(range(len(selected_individual)))
        count=0
        while count<5:
            #print("INSIDE BREED POPULATION")
            np.random.shuffle(mask)
            sample=random.sample(mask,2)
            parent1=selected_individual[sample[0]]
            parent2=selected_individual[sample[1]]
            if parent1!=parent2:
                break
            count+=1
        child1,child2=crossover(parent1,parent2)
        children.append(child1)
        children.append(child2)
    return children
        

def mutated_gene(gene,chromosome):
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


def three_nearest_neighbour_mutation(chromosome,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse):
	temp=copy.deepcopy(chromosome)
	for i in range(len(temp)):
		gene=temp[i]
		random_no=np.random.random()
		if random_no<mutation_probability:
			gene1=mutated_gene(gene,temp)
			temp[i]=gene1
		else:
			temp[i]=gene

	if tour_length_and_coverage(temp,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse)<tour_length_and_coverage(chromosome,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse):
		return temp 
	else:
		return chromosome



		       




'''
def mutate(individual,mutationrate,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse):
    temp=copy.deepcopy(individual)
    for swap_point in range(len(individual)):
        if(random.random()<mutationrate):
            swap_with=int(random.random()*len(individual))
            city1=individual[swap_point]
            city2=individual[swap_with]
            temp[swap_with]=city1
            temp[swap_point]=city2
    if tour_length_and_coverage(temp,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse)<tour_length_and_coverage(individual,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse):
        return temp
    else:
        return individual
            
   '''         




def mutate_population(population,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse):
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
        
    





def next_generation_without_mutation(population):
    selected_individual=[]
    selection_result=selection(rank_routes(population))
    for item in selection_result:
        selected_individual.append(population[item])
    children=breed_population(selected_individual)
    return children
    





def next_generation_with_mutation(population):
    selected_individual=[]
    selection_result=selection(rank_routes(population))
    for item in selection_result:
        selected_individual.append(population[item])
    children=breed_population(selected_individual)
    min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse=utility_min_max(children)

    mutated_population=mutate_population(children,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse)
    new_population=population+mutated_population

    

    ranked_new_population=rank_routes(new_population)
    
    
    indices=[item[0] for item in ranked_new_population]
    
    selected_indices=[indices[i] for i in range(population_size)]
    
    selected_population=[]

    for ind in selected_indices:
        selected_population.append(new_population[ind])

    
    #print("Selected population ")
    #print(selected_population)

    return selected_population
        
        





def initialisation(data):
    #f=open('result.txt',"a+")
    
   
    
    population=[]
    for i in range(population_size):
        no_of_cluster=np.random.randint(k_min,k_max+1)
        M, C = kmedoids.kMedoids(D,no_of_cluster)
       
        medoid=[]
        for item in M:
            medoid.append(data[item])
        if medoid not in population:
            population.append(medoid)
    progress=[]
    tl=[]
    min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse=utility_min_max(population)
    for item in population:
        l=tour_length_and_coverage(item,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse)
        tl.append(l)
    progress.append(min(tl))
    
        
    for i in range(no_of_generation_with_mutation):
        print("generation no with mutation ",i+1)
        population=next_generation_with_mutation(population)
        tl=[]
        min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse=utility_min_max(population)
        for item in population:
            l=tour_length_and_coverage(item,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse)
            tl.append(l)
        progress.append(min(tl))
   
    
        
    for i in range(no_of_generation_without_mutation):
        print("generation no without mutation",i+1)
        population=next_generation_without_mutation(population)
        tl=[]
        min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse=utility_min_max(population)
        for item in population:
            l=tour_length_and_coverage(item,min_tourLength,max_tourLength,min_coverageInverse,max_coverageInverse)
            tl.append(l)
        progress.append(min(tl))
        
    #f.write(str(min_tourLength)+" "+str(1/min_coverageInverse)+"\n")
    #f.close()

    
    print(progress)
    print("FINAL SOLUTION TO BE TAKEN ")
    print(min_tourLength,1/min_coverageInverse)
    plt.xlabel("Generation")
    plt.ylabel("w1*Tour_lengthCoverage + w2.coverage_inverse")
    plt.plot(progress)
    plt.show()





# All the program statements
# k_min is the minimum percentage of cluster
#k max is the maximum percentage of cluster
print("WSGA Based process")
k_min=165
k_max=198
radius=500
population_size=50
no_of_generation_with_mutation=150
no_of_generation_without_mutation=0
mutation_probability=0.1

print("radius ",radius,"k_min ",k_min,"k_max ",k_max)

with open('/content/drive/My Drive/Colab Notebooks/SJC818.txt') as f:
    data=[]
    index_of_data=[]
    for line in f:
        x,y=(line.strip('\n').split())
        data.append([int(x),int(y)])



    w1=0.5
    w2=0.5
    D=pairwise_distances(data,metric='euclidean')
    initialisation(data)
    




