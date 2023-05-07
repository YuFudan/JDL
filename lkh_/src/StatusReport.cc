#include "LKH.h"

void StatusReport(double EntryTime, const char *Suffix) {
    if (Penalty) {
        if (TruePenalty >= 0) {
            printff("Cost = " GainFormat "_" GainFormat "_" GainFormat, TruePenalty, CurrentPenalty, CurrentCost);
        } else {
            printff("Cost = " GainFormat "_" GainFormat, CurrentPenalty, CurrentCost);
        }
        if (Optimum != -INFINITY && Optimum != 0) {
            if (ProblemType != CCVRP && ProblemType != TRP &&
                ProblemType != MLP &&
                MTSPObjective != MINMAX && MTSPObjective != MINMAX_SIZE)
                printff(", Gap = %0.4f%%",
                        100.0 * (CurrentCost - Optimum) / Optimum);
            else
                printff(", Gap = %0.4f%%",
                        100.0 * (CurrentPenalty - Optimum) / Optimum);
        }
        printff(", Time = %0.6f sec. %s",
                fabs(GetTime() - EntryTime), Suffix);
    } else {
        printff("Cost = " GainFormat, CurrentCost);
        if (Optimum != -INFINITY && Optimum != 0)
            printff(", Gap = %0.4f%%", 100.0 * (CurrentCost - Optimum) / Optimum);
        printff(", Time = %0.6f sec.%s%s",
                fabs(GetTime() - EntryTime), Suffix,
                CurrentCost < Optimum ? "<" : CurrentCost == Optimum ? " =" : "");
    }
    printff("\n");
}
