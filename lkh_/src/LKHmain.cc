#include <stdlib.h>

#include "BIT.h"
#include "Genetic.h"
#include "LKH.h"

/*
 * This file contains the main function of the program.
 */

int main(int argc, char *argv[]) {
    GainType OldOptimum;
    double Time, LastTime;
    Node *N;
    int64_t i;
    Err = exit;
    /* Read the specification of the problem */
    ReadParameters(argc >= 2 ? argv[1] : nullptr);
    StartTime = LastTime = GetTime();
    MaxMatrixDimension = 20000;
    MergeWithTour = Recombination == IPT ? MergeWithTourIPT : MergeWithTourGPX2;
    CostTimeMatrix = NULL;
    CostDistanceMatrix = NULL;
    TruePenalty = -1;
    ReadProblem(ProblemFileName);
    if (SubproblemSize > 0) {
        if (DelaunayPartitioning)
            SolveDelaunaySubproblems();
        else if (KarpPartitioning)
            SolveKarpSubproblems();
        else if (KCenterPartitioning)
            SolveKCenterSubproblems();
        else if (KMeansPartitioning)
            SolveKMeansSubproblems();
        else if (RohePartitioning)
            SolveRoheSubproblems();
        else if (MoorePartitioning || SierpinskiPartitioning)
            SolveSFCSubproblems();
        else
            SolveTourSegmentSubproblems();
        return EXIT_SUCCESS;
    }
    AllocateStructures();
    if (ProblemType == TSPTW)
        TSPTW_Reduce();
    if (ProblemType == VRPB || ProblemType == VRPBTW)
        VRPB_Reduce();
    if (ProblemType == PDPTW)
        PDPTW_Reduce();
    CreateCandidateSet();
    InitializeStatistics();

    if (Norm != 0 || Penalty) {
        Norm = 9999;
        BestCost = INFINITY;
        BestPenalty = CurrentPenalty = INFINITY;
    } else {
        /* The ascent has solved the problem! */
        Optimum = BestCost = (GainType)LowerBound;
        UpdateStatistics(Optimum, GetTime() - LastTime);
        RecordBetterTour();
        RecordBestTour();
        CurrentPenalty = INFINITY;
        BestPenalty = CurrentPenalty = Penalty ? Penalty() : 0;
        WriteTour(OutputTourFileName, BestTour, BestCost);
        WriteTour(TourFileName, BestTour, BestCost);
        Runs = 0;
    }

    /* Find a specified number (Runs) of local optima */

    for (Run = 1; Run <= Runs; Run++) {
        LastTime = GetTime();
        if (LastTime - StartTime >= TimeLimit) {
            if (TraceLevel >= 1)
                printff("*** Time limit exceeded ***\n");
            break;
        }
        CurrentCost = FindTour(); /* using the Lin-Kernighan heuristic */
        if (MaxPopulationSize > 1 && !TSPTW_Makespan) {
            /* Genetic algorithm */
            int64_t i;
            for (i = 0; i < PopulationSize; i++) {
                GainType OldPenalty = CurrentPenalty;
                GainType OldCost = CurrentCost;
                CurrentCost = MergeTourWithIndividual(i);
                if (TraceLevel >= 1 &&
                    (CurrentPenalty < OldPenalty ||
                     (CurrentPenalty == OldPenalty && CurrentCost < OldCost))) {
                    if (CurrentPenalty)
                        printff("  Merged with %ld: CurrentCost = " GainFormat,
                                i + 1, CurrentCost);
                    else
                        printff("  Merged with %ld: CurrentCost = " GainFormat "_" GainFormat, i + 1, CurrentPenalty, CurrentCost);
                    if (Optimum != -INFINITY && Optimum != 0) {
                        if (ProblemType != CCVRP && ProblemType != TRP &&
                            ProblemType != MLP &&
                            MTSPObjective != MINMAX &&
                            MTSPObjective != MINMAX_SIZE)
                            printff(", Gap = %0.4f%%",
                                    100.0 * (CurrentCost - Optimum) / Optimum);
                        else
                            printff(", Gap = %0.4f%%",
                                    100.0 * (CurrentPenalty - Optimum) /
                                        Optimum);
                    }
                    printff("\n");
                }
            }
            if (!HasFitness(CurrentPenalty, CurrentCost)) {
                if (PopulationSize < MaxPopulationSize) {
                    AddToPopulation(CurrentPenalty, CurrentCost);
                    if (TraceLevel >= 1)
                        PrintPopulation();
                } else if (SmallerFitness(CurrentPenalty, CurrentCost,
                                          PopulationSize - 1)) {
                    i = ReplacementIndividual(CurrentPenalty, CurrentCost);
                    ReplaceIndividualWithTour(i, CurrentPenalty, CurrentCost);
                    if (TraceLevel >= 1)
                        PrintPopulation();
                }
            }
        } else if (Run > 1 && !TSPTW_Makespan)
            CurrentCost = MergeTourWithBestTour();
        if (CurrentPenalty < BestPenalty ||
            (CurrentPenalty == BestPenalty && CurrentCost < BestCost)) {
            BestPenalty = CurrentPenalty;
            BestCost = CurrentCost;
            RecordBetterTour();
            RecordBestTour();
            WriteTour(TourFileName, BestTour, BestCost);
        }
        OldOptimum = Optimum;
        if (!Penalty ||
            (MTSPObjective != MINMAX && MTSPObjective != MINMAX_SIZE)) {
            if (CurrentPenalty == 0 && CurrentCost < Optimum)
                Optimum = CurrentCost;
        } else if (CurrentPenalty < Optimum)
            Optimum = CurrentPenalty;
        if (Optimum < OldOptimum) {
            printff("*** New OPTIMUM = " GainFormat " ***\n", Optimum);
            if (FirstNode->InputSuc) {
                Node *N = FirstNode;
                while ((N = N->InputSuc = N->Suc) != FirstNode)
                    ;
            }
        }
        Time = fabs(GetTime() - LastTime);
        UpdateStatistics(CurrentCost, Time);
        if (TraceLevel >= 1 && CurrentCost != INFINITY) {
            printff("Run %ld: ", Run);
            StatusReport(LastTime, "");
            printff("\n");
        }
        if (StopAtOptimum && MaxPopulationSize >= 1) {
            if (ProblemType != CCVRP && ProblemType != TRP &&
                        ProblemType != MLP &&
                        MTSPObjective != MINMAX &&
                        MTSPObjective != MINMAX_SIZE
                    ? CurrentPenalty == 0 && CurrentCost == Optimum
                    : CurrentPenalty == Optimum) {
                Runs = Run;
                break;
            }
        }
        if (PopulationSize >= 2 &&
            (PopulationSize == MaxPopulationSize ||
             Run >= 2 * MaxPopulationSize) &&
            Run < Runs) {
            Node *N;
            int64_t Parent1, Parent2;
            Parent1 = LinearSelection(PopulationSize, 1.25);
            do
                Parent2 = LinearSelection(PopulationSize, 1.25);
            while (Parent2 == Parent1);
            ApplyCrossover(Parent1, Parent2);
            N = FirstNode;
            do {
                if (ProblemType != HCP && ProblemType != HPP) {
                    int64_t d = C(N, N->Suc);
                    AddCandidate(N, N->Suc, d, INT_MAX);
                    AddCandidate(N->Suc, N, d, INT_MAX);
                }
                N = N->InitialSuc = N->Suc;
            } while (N != FirstNode);
        }
        SRandom(++Seed);
    }
    PrintStatistics();
    if (Salesmen > 1) {
        if (Dimension == Dim_salesman) {
            for (i = 1; i <= Dimension; i++) {
                N = &NodeSet[BestTour[i - 1]];
                (N->Suc = &NodeSet[BestTour[i]])->Pred = N;
            }
        } else {
            for (i = 1; i <= Dim_salesman; i++) {
                Node *N1 = &NodeSet[BestTour[i - 1]];
                Node *N2 = &NodeSet[BestTour[i]];
                Node *M1 = &NodeSet[N1->Id + Dim_salesman];
                Node *M2 = &NodeSet[N2->Id + Dim_salesman];
                (M1->Suc = N1)->Pred = M1;
                (N1->Suc = M2)->Pred = N1;
                (M2->Suc = N2)->Pred = M2;
            }
        }
        CurrentPenalty = BestPenalty;
        MTSP_Report(BestPenalty, BestCost);
        MTSP_WriteSolution(MTSPSolutionFileName, BestPenalty, BestCost);
        SINTEF_WriteSolution(SINTEFSolutionFileName, BestCost);
    }
    if (ProblemType == ACVRP ||
        ProblemType == BWTSP ||
        ProblemType == CCVRP ||
        ProblemType == CTSP ||
        ProblemType == CVRP ||
        ProblemType == CVRPTW ||
        ProblemType == MLP ||
        ProblemType == M_PDTSP ||
        ProblemType == M1_PDTSP ||
        MTSPObjective != -1 ||
        ProblemType == ONE_PDTSP ||
        ProblemType == OVRP ||
        ProblemType == PDTSP ||
        ProblemType == PDTSPL ||
        ProblemType == PDPTW ||
        ProblemType == RCTVRP ||
        ProblemType == RCTVRPTW ||
        ProblemType == SOP ||
        ProblemType == TRP ||
        ProblemType == TSPTW ||
        ProblemType == VRPB ||
        ProblemType == VRPBTW || ProblemType == VRPSPDTW) {
        puts("Best solution:");
        CurrentPenalty = BestPenalty;
        SOP_Report(BestCost);
    }
    printff("\n");
    return EXIT_SUCCESS;
}
