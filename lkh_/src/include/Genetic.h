#ifndef _GENETIC_H
#define _GENETIC_H

/*
 * This header specifies the interface for the genetic algorithm part of LKH.
 */

#include "GainType.h"

#define SmallerFitness(Penalty, Cost, i) \
    (((Penalty) < PenaltyFitness[i]) ||  \
     ((Penalty) == PenaltyFitness[i] && (Cost) < Fitness[i]))

#define LargerFitness(Penalty, Cost, i) \
    (((Penalty) > PenaltyFitness[i]) || \
     ((Penalty) == PenaltyFitness[i] && (Cost) > Fitness[i]))

typedef void (*CrossoverFunction)();

extern int64_t MaxPopulationSize; /* The maximum size of the population */
extern int64_t PopulationSize;    /* The current size of the population */
extern CrossoverFunction Crossover;
extern int64_t **Population;         /* Array of individuals (solution tours) */
extern GainType *PenaltyFitness; /* The fitness (tour penalty) of each
                                    individual */
extern GainType *Fitness;        /* The fitness (tour cost) of each individual */

void AddToPopulation(GainType Penalty, GainType Cost);
void ApplyCrossover(int64_t i, int64_t j);
void FreePopulation();
int64_t HasFitness(GainType Penalty, GainType Cost);
int64_t LinearSelection(int64_t Size, double Bias);
GainType MergeTourWithIndividual(int64_t i);
void PrintPopulation();
void ReplaceIndividualWithTour(int64_t i, GainType Penalty, GainType Cost);
int64_t ReplacementIndividual(GainType Penalty, GainType Cost);

void ERXT();

#endif
