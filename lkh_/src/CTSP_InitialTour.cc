#include "LKH.h"

/* The CTSP_InitialTour function computes an initial tour for a
 * colored TSP.
 */

GainType CTSP_InitialTour() {
    Node *N, *NextN;
    GainType Cost;
    int64_t Set;
    double EntryTime = GetTime();

    if (TraceLevel >= 1)
        printff("CTSP = ");
    ASSERT(!Asymmetric);
    for (Set = 2; Set <= Salesmen; Set++)
        Follow(&NodeSet[Dim_original + Set - 1],
               Set == 2 ? Depot : &NodeSet[Dim_original + Set - 2]);
    N = Depot;
    do {
        NextN = N->Suc;
        if (N->DepotId == 0) {
            Set = N->Color != 0 ? N->Color : Random() % Salesmen + 1;
            Follow(N, Set == 1 ? Depot : &NodeSet[Dim_original + Set - 1]);
        }
    } while ((N = NextN) != Depot);
    Cost = 0;
    N = FirstNode;
    do
        Cost += C(N, N->Suc) - N->Pi - N->Suc->Pi;
    while ((N = N->Suc) != FirstNode);
    CurrentPenalty = INFINITY;
    CurrentPenalty = Penalty();
    if (TraceLevel >= 1) {
        if (Salesmen > 1 || ProblemType == SOP)
            printff(GainFormat "_" GainFormat, CurrentPenalty, Cost);
        else
            printff(GainFormat, Cost);
        if (Optimum != -INFINITY && Optimum != 0)
            printff(", Gap = %0.2f%%", 100.0 * (Cost - Optimum) / Optimum);
        printff(", Time = %0.6f sec.\n", fabs(GetTime() - EntryTime));
    }
    return Cost;
}
