#include "LKH.h"

void SINTEF_WriteSolution(char *FileName, GainType Cost) {
    FILE *ResultFile;
    Node *N, *NextN;
    char *FullFileName;
    int64_t Route, Forward;
    time_t Now;

    if (FileName == 0)
        return;
    FullFileName = FullName(FileName, Cost);
    Now = time(&Now);
    if (TraceLevel >= 1)
        printff("Writing SINTEF_SOLUTION_FILE: \"%s\" ... ",
                FullFileName);
    ResultFile = fopen(FullFileName, "w");
    fprintf(ResultFile, "Instance name : %s\n", Name);
    fprintf(ResultFile, "Authors       : Keld Helsgaun\n");
    fprintf(ResultFile, "Date          : %s", ctime(&Now));
    fprintf(ResultFile,
            "Reference     : "
            "http://webhotel4.ruc.dk/~keld/research/LKH-3\n");
    fprintf(ResultFile, "Solution\n");
    N = Depot;
    Forward = N->Suc->Id != N->Id + Dim_salesman;
    Route = 0;
    do {
        Route++;
        fprintf(ResultFile, "Route %ld : ", Route);
        do {
            if (N->Id <= Dim_original && N != Depot)
                fprintf(ResultFile, "%ld ", N->Id - 1);
            NextN = Forward ? N->Suc : N->Pred;
            if (NextN->Id > Dim_salesman)
                NextN = Forward ? NextN->Suc : NextN->Pred;
            N = NextN;
        } while (N->DepotId == 0);
        fprintf(ResultFile, "\n");
    } while (N != Depot);
    fclose(ResultFile);
    if (TraceLevel >= 1)
        printff("done\n");
}
