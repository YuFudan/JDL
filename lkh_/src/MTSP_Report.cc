#include "LKH.h"
#include "Segment.h"

void MTSP_Report(GainType Penalty, GainType Cost) {
    GainType MinCost = INFINITY, MaxCost = -INFINITY;
    int64_t MinSize = INT_MAX, MaxSize = 0;
    Node *N = Depot, *NextN;
    bool Forward = SUCC(Depot)->Id != Depot->Id + Dim_salesman;
    int64_t SalesmenUsed = 0;

    do {
        GainType Cost = 0;
        int64_t Size = -1;
        do {
            NextN = Forward ? SUCC(N) : PREDD(N);
            Cost += N->CD[NextN->Index];
            if (NextN->Id > Dim_salesman)
                NextN = Forward ? SUCC(NextN) : PREDD(NextN);
            Size++;
        } while ((N = NextN)->DepotId == 0);
        if (Cost < MinCost)
            MinCost = Cost;
        if (Cost > MaxCost)
            MaxCost = Cost;
        if (Size < MinSize)
            MinSize = Size;
        if (Size > MaxSize)
            MaxSize = Size;
        if (Size > 0)
            SalesmenUsed++;
    } while (N != Depot);
    if (MTSPMinSize == 0)
        printff("  Salesmen/vehicles used = %ld\n", SalesmenUsed);
    printff("  Size.min = %ld, Size.max = %ld, Penalty = " GainFormat "\n",
            MinSize, MaxSize, Penalty);
    printff("  Cost.min = " GainFormat ", Cost.max = " GainFormat,
            MinCost, MaxCost);
    printff(", Cost.sum = " GainFormat "\n", Cost);
}
