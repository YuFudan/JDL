#include "LKH.h"

static int64_t TrialsMin, TrialsMax, TrialSum, Successes;
static GainType CostMin, CostMax, CostSum;
static GainType PenaltyMin, PenaltyMax, PenaltySum;
static double TimeMin, TimeMax, TimeSum;

void InitializeStatistics() {
    TrialSum = Successes = 0;
    CostSum = 0;
    TimeSum = 0.0;
    TrialsMin = INT_MAX;
    TrialsMax = 0;
    TimeMin = INFINITY;
    TimeMax = 0;
    CostMin = INFINITY;
    CostMax = -INFINITY;
    PenaltySum = 0;
    PenaltyMin = INFINITY;
    PenaltyMax = -INFINITY;
}

void UpdateStatistics(GainType Cost, double Time) {
    if (Trial < TrialsMin)
        TrialsMin = Trial;
    if (Trial > TrialsMax)
        TrialsMax = Trial;
    TrialSum += Trial;
    if (Cost < CostMin)
        CostMin = Cost;
    if (Cost > CostMax)
        CostMax = Cost;
    CostSum += Cost;
    if (ProblemType != CCVRP && ProblemType != TRP &&
        ProblemType != MLP &&
        MTSPObjective != MINMAX && MTSPObjective != MINMAX_SIZE) {
        if (CurrentPenalty == 0 && Cost <= Optimum)
            Successes++;
    } else if (CurrentPenalty <= Optimum)
        Successes++;
    if (CurrentPenalty < PenaltyMin)
        PenaltyMin = CurrentPenalty;
    if (CurrentPenalty > PenaltyMax)
        PenaltyMax = CurrentPenalty;
    PenaltySum += CurrentPenalty;
    if (Time < TimeMin)
        TimeMin = Time;
    if (Time > TimeMax)
        TimeMax = Time;
    TimeSum += Time;
}

void PrintStatistics() {
    int64_t _Runs = Run - 1, _TrialsMin = TrialsMin;
    double _TimeMin = TimeMin;
    GainType _Optimum = Optimum;

    printff("Successes/Runs = %ld/%ld \n", Successes, Runs);
    if (_Runs == 0)
        _Runs = 1;
    if (_TrialsMin > TrialsMax)
        _TrialsMin = 0;
    if (_TimeMin > TimeMax)
        _TimeMin = 0;
    if (ProblemType != CCVRP && ProblemType != TRP &&
        ProblemType != MLP &&
        MTSPObjective != MINMAX && MTSPObjective != MINMAX_SIZE &&
        CostMin <= CostMax && CostMin != INFINITY) {
        printff("Cost.min = " GainFormat
                ", Cost.avg = %0.2f, "
                "Cost.max = " GainFormat "\n",
                CostMin, (double)CostSum / _Runs, CostMax);
        if (_Optimum == -INFINITY)
            _Optimum = BestCost;
        if (_Optimum != 0)
            printff(
                "Gap.min = %0.4f%%, Gap.avg = %0.4f%%, "
                "Gap.max = %0.4f%%\n",
                100.0 * (CostMin - _Optimum) / _Optimum,
                100.0 * ((double)CostSum / _Runs - _Optimum) / _Optimum,
                100.0 * (CostMax - _Optimum) / _Optimum);
        if (Penalty && PenaltyMin != INFINITY)
            printff("Penalty.min = " GainFormat
                    ", Penalty.avg = %0.2f, "
                    "Penalty.max = " GainFormat "\n",
                    PenaltyMin, (double)PenaltySum / _Runs, PenaltyMax);
    } else if (Penalty && PenaltyMin != INFINITY) {
        printff("Penalty.min = " GainFormat
                ", Penalty.avg = %0.2f, "
                "Penalty.max = " GainFormat "\n",
                PenaltyMin, (double)PenaltySum / _Runs, PenaltyMax);
        if (_Optimum == -INFINITY)
            _Optimum = BestPenalty;
        if (_Optimum != 0)
            printff(
                "Gap.min = %0.4f%%, Gap.avg = %0.4f%%, "
                "Gap.max = %0.4f%%\n",
                100.0 * (PenaltyMin - _Optimum) / _Optimum,
                100.0 * ((double)PenaltySum / _Runs - _Optimum) / _Optimum,
                100.0 * (PenaltyMax - _Optimum) / _Optimum);
    }
    printff("Trials.min = %ld, Trials.avg = %0.1f, Trials.max = %ld\n",
            _TrialsMin, 1.0 * TrialSum / _Runs, TrialsMax);
    printff(
        "Time.min = %0.2f sec., Time.avg = %0.2f sec., "
        "Time.max = %0.2f sec.\n",
        fabs(_TimeMin), fabs(TimeSum) / _Runs, fabs(TimeMax));
    printff("Time.total = %0.2f sec.\n", GetTime() - StartTime);
}
