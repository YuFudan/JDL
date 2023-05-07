#include "LKH.h"

void MTSP_WriteSolution(char *FileName, GainType Penalty, GainType Cost) {
    FILE *SolutionFile;
    Node *N, *NextN;
    int64_t Size, Forward;
    char *FullFileName;
    GainType Sum;

    if (FileName == 0)
        return;
    FullFileName = FullName(FileName, Cost);
    if (TraceLevel >= 1)
        printff("Writing MTSP_SOLUTION_FILE: \"%s\" ... ", FullFileName);
    ASSERT((SolutionFile = fopen(FullFileName, "w")));
    fprintf(SolutionFile, "%s, Cost: " GainFormat "_" GainFormat "\n",
            Name, Penalty, Cost);
    fprintf(SolutionFile, "The tours traveled by the %ld salesmen are:\n",
            Salesmen);
    N = Depot;
    Forward = N->Suc->Id != N->Id + Dim_salesman;
    do {
        Sum = 0;
        Size = -1;
        do {
            fprintf(SolutionFile, "%ld ", N->Id <= Dim_original ? N->Id : Depot->Id);
            NextN = Forward ? N->Suc : N->Pred;
            Sum += N->CD[NextN->Index];
            Size++;
            if (NextN->Id > Dim_salesman)
                NextN = Forward ? NextN->Suc : NextN->Pred;
            N = NextN;
        } while (N->DepotId == 0);
        if (N->DepotId <= ExternalSalesmen)
            fprintf(SolutionFile, "(#%ld)  Cost: " GainFormat "\n",
                    Size, Sum);
        else
            fprintf(SolutionFile, "%ld (#%ld)  Cost: " GainFormat "\n",
                    Depot->Id, Size, Sum);
    } while (N != Depot);
    fclose(SolutionFile);
    if (TraceLevel >= 1)
        printff("done\n");
}
