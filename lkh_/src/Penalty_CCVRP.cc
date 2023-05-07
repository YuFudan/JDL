#include "LKH.h"
#include "Segment.h"

GainType Penalty_CCVRP() {
    static Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    GainType DemandSum, DistanceSum, MaxDistanceSum = 0, P = 0;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;

    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > Dim_salesman)
        StartRoute -= Dim_salesman;
    N = StartRoute;
    do {
        CurrentRoute = N;
        DemandSum = 0;
        DistanceSum = 0;
        do {
            if (N->Id <= Dim_original && N != Depot) {
                if ((DemandSum += N->Demand) > Capacity)
                    P += 10000 * (DemandSum - Capacity);
                if (P > CurrentPenalty ||
                    (P == CurrentPenalty && CurrentGain <= 0)) {
                    StartRoute = CurrentRoute;
                    return CurrentPenalty + (CurrentGain > 0);
                }
            }
            NextN = Forward ? SUCC(N) : PREDD(N);
            if (NextN->DepotId == 0) {
                DistanceSum += N->CD[NextN->Index];
                if (MTSPObjective != MINMAX) {
                    P += DistanceSum;
                    if (P > CurrentPenalty ||
                        (P == CurrentPenalty && CurrentGain <= 0)) {
                        StartRoute = CurrentRoute;
                        return CurrentPenalty + (CurrentGain > 0);
                    }
                } else if (DistanceSum > MaxDistanceSum) {
                    MaxDistanceSum = DistanceSum;
                    P += MaxDistanceSum;
                    if (P > CurrentPenalty ||
                        (P == CurrentPenalty && CurrentGain <= 0)) {
                        StartRoute = CurrentRoute;
                        return CurrentPenalty + (CurrentGain > 0);
                    }
                    P -= MaxDistanceSum;
                }
            }
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while (N->DepotId == 0);
    } while (N != StartRoute);
    return P + MaxDistanceSum;
}
