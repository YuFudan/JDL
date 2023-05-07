#include "LKH.h"
#include "Segment.h"

static int64_t *Stack;

GainType Penalty_PDTSPL() {
    static Node *StartRoute = 0;
    Node *N, *NextN, *CurrentRoute;
    GainType P = 0;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;
    int64_t StackTop;

    if (!Stack)
        Stack = new int64_t[Dim_original]();
    if (!StartRoute)
        StartRoute = Depot;
    if (StartRoute->Id > Dim_salesman)
        StartRoute -= Dim_salesman;
    N = StartRoute;
    do {
        CurrentRoute = N;
        StackTop = -1;
        do {
            if (N->Id <= Dim_original && N != Depot) {
                if (N->Pickup) {
                    if (StackTop == -1 || Stack[StackTop--] != N->Id)
                        P++;
                    if (P > CurrentPenalty ||
                        (P == CurrentPenalty && CurrentGain <= 0)) {
                        StartRoute = CurrentRoute;
                        return CurrentPenalty + (CurrentGain > 0);
                    }
                } else if (N->Delivery)
                    Stack[++StackTop] = N->Delivery;
            }
            NextN = Forward ? SUCC(N) : PREDD(N);
            N = Forward ? SUCC(NextN) : PREDD(NextN);
        } while (N->DepotId == 0);
        P += StackTop + 1;
    } while (N != StartRoute);
    return P;
}
