#include "LKH.h"
#include "Segment.h"

GainType Penalty_VRPSPDTW() {
    static Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    GainType Duration, P = 0;
    int64_t Load, DeliverySum, PickupSum;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;

    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > Dim_salesman)
        StartRoute -= Dim_salesman;
    if (SingleCourier) {
        N = StartRoute = Depot;
        Duration = Depot->Earliest;
    } else {
        N = StartRoute;
    }
    do {
        CurrentRoute = N;
        DeliverySum = PickupSum = 0;
        do {
            if (N->Id <= Dim_original && N != Depot) {
                DeliverySum += N->Delivery;
                PickupSum += N->Pickup;
            }
            NextN = Forward ? SUCC(N) : PREDD(N);
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while (N->DepotId == 0);
        if (DeliverySum > Capacity)
            P += DeliverySum - Capacity;
        if (PickupSum > Capacity)
            P += PickupSum - Capacity;
        if (P > CurrentPenalty ||
            (P == CurrentPenalty && CurrentGain <= 0)) {
            StartRoute = CurrentRoute;
            return CurrentPenalty + (CurrentGain > 0);
        }
        if (!SingleCourier)
            Duration = Depot->Earliest;
        Load = DeliverySum;
        N = CurrentRoute;
        do {
            if (N->Id <= Dim_original && N != Depot) {
                Load -= N->Delivery;
                Load += N->Pickup;
                if (Load > Capacity)
                    P += Load - Capacity;
                if (Duration < N->Earliest)
                    Duration = N->Earliest;
                else if (Duration > N->Latest)
                    P += Duration - N->Latest;
                // else if (waiting_penalty>0)
                if (P > CurrentPenalty) {
                    StartRoute = CurrentRoute;
                    return CurrentPenalty + 1;
                }
                Duration += N->ServiceTime;
            }
            NextN = Forward ? SUCC(N) : PREDD(N);
            Duration += NextN->Index[CostTimeMatrix ? N->CT : N->CD];
            if (DistanceLimit != 0 && Duration > DistanceLimit)
                P += Duration - DistanceLimit;
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while (N->DepotId == 0);
        if (Duration > Depot->Latest &&
            (P += Duration - Depot->Latest) > CurrentPenalty) {
            StartRoute = CurrentRoute;
            return CurrentPenalty + 1;
        }
    } while (N != StartRoute);
    return P;
}
