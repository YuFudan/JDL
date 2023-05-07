#include "Genetic.h"

#include "LKH.h"

/*
 * The AddToPopulation function adds the current tour as an individual to
 * the population. The fitness of the individual is set equal to the cost
 * of the tour. The population is kept sorted in increasing fitness order.
 */

void AddToPopulation(GainType Penalty, GainType Cost) {
    int64_t i, *P;
    Node *N;

    if (!Population) {
        Population = new int64_t *[MaxPopulationSize]();
        for (i = 0; i < MaxPopulationSize; i++)
            Population[i] = new int64_t[1 + Dimension]();
        PenaltyFitness =
            new GainType[MaxPopulationSize]();
        Fitness =
            new GainType[MaxPopulationSize]();
    }
    for (i = PopulationSize;
         i >= 1 && SmallerFitness(Penalty, Cost, i - 1); i--) {
        PenaltyFitness[i] = PenaltyFitness[i - 1];
        Fitness[i] = Fitness[i - 1];
        P = Population[i];
        Population[i] = Population[i - 1];
        Population[i - 1] = P;
    }
    PenaltyFitness[i] = Penalty;
    Fitness[i] = Cost;
    P = Population[i];
    N = FirstNode;
    i = 1;
    do
        P[i++] = N->Id;
    while ((N = N->Suc) != FirstNode);
    P[0] = P[Dimension];
    PopulationSize++;
}

/*
 * The ApplyCrossover function applies a specified crossover operator to two
 * individuals.
 */

void ApplyCrossover(int64_t i, int64_t j) {
    int64_t *Pi, *Pj, k;

    Pi = Population[i];
    Pj = Population[j];
    for (k = 1; k <= Dimension; k++) {
        NodeSet[Pi[k - 1]].Suc = &NodeSet[Pi[k]];
        NodeSet[Pj[k - 1]].Next = &NodeSet[Pj[k]];
    }
    if (TraceLevel >= 1)
        printff("Crossover(%ld,%ld)\n", i + 1, j + 1);
    /* Apply the crossover operator */
    Crossover();
    if (ProblemType == SOP)
        SOP_RepairTour();
}

#define Free(s)  \
    {            \
        free(s); \
        s = 0;   \
    }

/*
 * The FreePopulation function frees the memory space allocated to the
 * population.
 */

void FreePopulation() {
    if (Population) {
        int64_t i;
        for (i = 0; i < MaxPopulationSize; i++)
            Free(Population[i]);
        Free(Population);
        Free(PenaltyFitness);
        Free(Fitness);
    }
    PopulationSize = 0;
}

/*
 * The HasFitness function returns 1 if the population contains an
 * individual with fitness equal to a given tour cost; otherwise 0.
 *
 * Since the population is sorted in fitness order the test may be
 * made by binary search.
 */

int64_t HasFitness(GainType Penalty, GainType Cost) {
    int64_t Low = 0, High = PopulationSize - 1;
    while (Low < High) {
        int64_t Mid = (Low + High) / 2;
        if (LargerFitness(Penalty, Cost, Mid))
            Low = Mid + 1;
        else
            High = Mid;
    }
    return High >= 0 && PenaltyFitness[High] == Penalty &&
           Fitness[High] == Cost;
}

/*
 * Random01 is an auxiliary function for computing a random double number
 * in the range [0;1).
 */

static double Random01() {
    return ((double)Random()) / INT_MAX;
}

/*
 * The LinearSelection function is used to select an individual with
 * random linear bias towards the best members of the population.
 * The parameter Bias is a number between 1.0 and 2.0.
 *
 * See
 *     Darrell Whitley,
 *     The GENITOR algorithm and selection pressure:
 *     Why rank-based allocation of reproductive trials is best.
 *     Proceedings of the Third International Conference on Genetic Algorithms,
 *     1989.
 */

int64_t LinearSelection(int64_t Size, double Bias) {
    return (int64_t)(Size *
                     (Bias -
                      sqrt((Bias * Bias - 4 * (Bias - 1) * Random01()))) /
                     2 / (Bias - 1));
}

/*
 * The MergeTourWithIndividual function attempts to find a short tour by
 * merging the current tour with a specified inddividual of the population.
 * The merging algorithm is the iterative partial transcription algrithm
 * described by Mobius, Freisleben, Merz and Schreiber.
 */

GainType MergeTourWithIndividual(int64_t i) {
    int64_t *Pi, k;

    ASSERT(i >= 0 && i < PopulationSize);
    Pi = Population[i];
    for (k = 1; k <= Dimension; k++)
        NodeSet[Pi[k - 1]].Next = &NodeSet[Pi[k]];
    return MergeWithTour();
}

/*
 * The PrintPopulation function prints the cost and gap to optimum for
 * each individual of the population.
 */

void PrintPopulation() {
    int64_t i;
    printff("Population:\n");
    for (i = 0; i < PopulationSize; i++) {
        printff("%3d: ", i + 1);
        if (Penalty)
            printff(GainFormat "_" GainFormat,
                    PenaltyFitness[i], Fitness[i]);
        else
            printff(GainFormat, Fitness[i]);
        if (Optimum != -INFINITY && Optimum != 0) {
            if (ProblemType != CCVRP && ProblemType != TRP &&
                ProblemType != MLP &&
                MTSPObjective != MINMAX &&
                MTSPObjective != MINMAX_SIZE)
                printff(", Gap = %0.4f%%",
                        100.0 * (Fitness[i] - Optimum) / Optimum);
            else
                printff(", Gap = %0.4f%%",
                        100.0 * (PenaltyFitness[i] - Optimum) / Optimum);
        }
        printff("\n");
    }
}

/*
 * The ReplaceIndividualWithTour function replaces a given individual in
 * the population by an indidual that represents the current tour.
 * The population is kept sorted in increasing fitness order.
 */

void ReplaceIndividualWithTour(int64_t i, GainType Penalty, GainType Cost) {
    int64_t j, *P;
    Node *N;

    ASSERT(i >= 0 && i < PopulationSize);
    PenaltyFitness[i] = Penalty;
    Fitness[i] = Cost;
    P = Population[i];
    N = FirstNode;
    for (j = 1; j <= Dimension; j++) {
        P[j] = N->Id;
        N = N->Suc;
    }
    P[0] = P[Dimension];
    while (i >= 1 && SmallerFitness(Penalty, Cost, i - 1)) {
        PenaltyFitness[i] = PenaltyFitness[i - 1];
        Fitness[i] = Fitness[i - 1];
        Population[i] = Population[i - 1];
        i--;
    }
    PenaltyFitness[i] = Cost;
    Fitness[i] = Cost;
    Population[i] = P;
    while (i < PopulationSize - 1 && LargerFitness(Penalty, Cost, i + 1)) {
        PenaltyFitness[i] = PenaltyFitness[i + 1];
        Fitness[i] = Fitness[i + 1];
        Population[i] = Population[i + 1];
        i++;
    }
    PenaltyFitness[i] = Penalty;
    Fitness[i] = Cost;
    Population[i] = P;
}

/*
 * The DistanceToIndividual returns the number of different edges between
 * the tour (given by OldSuc) and individual i.
 */

static int64_t DistanceToIndividual(int64_t i) {
    int64_t Count = 0, j, *P = Population[i];
    Node *N;

    for (j = 0; j < Dimension; j++) {
        N = &NodeSet[P[j]];
        (N->Next = &NodeSet[P[j + 1]])->Prev = N;
    }
    N = FirstNode;
    do
        if (N->OldSuc != N->Next && N->OldSuc != N->Prev)
            Count++;
    while ((N = N->OldSuc) != FirstNode);
    return Count;
}

/*
 * The ReplacementIndividual function returns the individual to be
 * replaced with the current tour. The function implements the
 * replacement strategy (CD/RW) proposed in
 *
 *      M. Lozano, F. Herrera, and J. R. Cano,
 *      Replacement strategies to preserve useful diversity in
 *      steady-state genetic algorithms.
 *      Information Sciences 178 (2008) 4421–4433.
 */

int64_t ReplacementIndividual(GainType Penalty, GainType Cost) {
    int64_t i, j, d, *P;
    int64_t MinDist = INT_MAX, CMin = PopulationSize - 1;
    Node *N = FirstNode;
    while ((N = N->OldSuc = N->Suc) != FirstNode)
        ;
    for (i = PopulationSize - 1;
         i >= 0 && SmallerFitness(Penalty, Cost, i); i--) {
        if ((d = DistanceToIndividual(i)) < MinDist) {
            CMin = i;
            MinDist = d;
        }
    }
    if (CMin == PopulationSize - 1)
        return CMin;
    P = Population[CMin];
    for (j = 0; j < Dimension; j++)
        NodeSet[P[j]].OldSuc = &NodeSet[P[j + 1]];
    for (i = 0; i < PopulationSize; i++)
        if (i != CMin && (d = DistanceToIndividual(i)) <= MinDist)
            return PopulationSize - 1;
    return CMin;
}
