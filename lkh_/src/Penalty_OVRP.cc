#include "LKH.h"
#include "Segment.h"

GainType Penalty_OVRP() {
    static Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    GainType DemandSum, DistanceSum, P = 0;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;

    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > Dim_salesman)
        StartRoute -= Dim_salesman;
    N = StartRoute;
    do {
        CurrentRoute = N;
        DistanceSum = DemandSum = 0;
        do {
            if (N->Id <= Dim_original && N != Depot) {
                if ((DemandSum += N->Demand) > Capacity)
                    P += DemandSum - Capacity;
                if (DistanceSum < N->Earliest)
                    DistanceSum = N->Earliest;
                if (DistanceSum > N->Latest)
                    P += DistanceSum - N->Latest;
                if (P > CurrentPenalty ||
                    (P == CurrentPenalty && CurrentGain <= 0)) {
                    StartRoute = CurrentRoute;
                    return CurrentPenalty + (CurrentGain > 0);
                }
                DistanceSum += N->ServiceTime;
            }
            NextN = Forward ? SUCC(N) : PREDD(N);
            if (DistanceLimit != INFINITY)
                DistanceSum += N->CD[NextN->Index];
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while (N->DepotId == 0);
        if (DistanceSum > DistanceLimit &&
            ((P += DistanceSum - DistanceLimit) > CurrentPenalty ||
             (P == CurrentPenalty && CurrentGain <= 0))) {
            StartRoute = CurrentRoute;
            return CurrentPenalty + (CurrentGain > 0);
        }
    } while (N != StartRoute);
    return P;
}
