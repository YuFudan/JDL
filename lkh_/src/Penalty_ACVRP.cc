#include "LKH.h"
#include "Segment.h"

GainType Penalty_ACVRP() {
    static Node *StartRoute = 0;
    Node *N, *CurrentRoute;
    GainType DemandSum, P = 0;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;

    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > Dim_salesman)
        StartRoute -= Dim_salesman;
    N = StartRoute;
    do {
        CurrentRoute = N;
        DemandSum = 0;
        do {
            if (N->Id <= Dim_original && N != Depot) {
                if ((DemandSum += N->Demand) > Capacity)
                    P += DemandSum - Capacity;
                if (P > CurrentPenalty ||
                    (P == CurrentPenalty && CurrentGain <= 0)) {
                    StartRoute = CurrentRoute;
                    return CurrentPenalty + (CurrentGain > 0);
                }
            }
            N = Forward ? SUCC(N) : PREDD(N);
        } while (N->DepotId == 0);
    } while (N != StartRoute);
    return P;
}
