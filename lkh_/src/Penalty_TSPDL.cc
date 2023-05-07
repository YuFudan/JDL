#include "LKH.h"
#include "Segment.h"

static int64_t DemandSum;

GainType Penalty_TSPDL() {
    Node *N = Depot, *NextN;
    GainType P = 0;
    int64_t Draft;
    bool Forward = SUCC(N)->Id != N->Id + Dim_salesman;

    if (DemandSum == 0) {
        do {
            if (N->Id <= Dim_salesman)
                DemandSum += N->Demand;
            NextN = Forward ? SUCC(N) : PREDD(N);
        } while ((N = NextN) != Depot);
    }
    Draft = DemandSum;
    do {
        if (N->Id <= Dim_salesman) {
            if (Draft > N->DraftLimit &&
                (P += Draft - N->DraftLimit) > CurrentPenalty)
                return CurrentPenalty + 1;
            Draft -= N->Demand;
            NextN = Forward ? SUCC(N) : PREDD(N);
        }
        NextN = Forward ? SUCC(N) : PREDD(N);
    } while ((N = NextN) != Depot);
    return P;
}
