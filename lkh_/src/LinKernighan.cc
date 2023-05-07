#include <initializer_list>

#include "BIT.h"
#include "Hashing.h"
#include "LKH.h"
#include "Segment.h"
#include "Sequence.h"

/*
 * The LinKernighan function seeks to improve a tour by sequential
 * and non-sequential edge exchanges.
 *
 * The function returns the cost of the resulting tour.
 */

GainType LinKernighan() {
    double EntryTime = GetTime();

    Reversed = false;
    FirstActive = LastActive = nullptr;
    Swaps = 0;
    CurrentCost = 0;
    Hash = 0;

    // 初始化segment结构
    // 计算初始回路的Cost和Hash
    // 标记所有结点为Active
    {
        auto *S = FirstSegment;
        int64_t i = 0;
        do {
            S->Size = 0;
            S->Rank = ++i;
            S->Reversed = 0;
            S->First = S->Last = 0;
        } while ((S = S->Suc) != FirstSegment);
        auto *SS = FirstSSegment;
        i = 0;
        do {
            SS->Size = 0;
            SS->Rank = ++i;
            SS->Reversed = 0;
            SS->First = SS->Last = 0;
        } while ((SS = SS->Suc) != FirstSSegment);
        i = 0;
        auto *t1 = FirstNode;
        do {
            auto *t2 = t1->OldSuc = t1->Suc;
            t1->OldPred = t1->Pred;
            t1->Rank = ++i;
            CurrentCost += (t1->SucCost = t2->PredCost = C(t1, t2)) - t1->Pi - t2->Pi;
            Hash ^= Rand[t1->Id] * Rand[t2->Id];
            t1->Cost = INT_MAX;
            for (auto *Nt1 = t1->CandidateSet.data(); (t2 = Nt1->To); Nt1++)
                if (t2 != t1->Pred && t2 != t1->Suc && Nt1->Cost < t1->Cost)
                    t1->Cost = Nt1->Cost;
            t1->Parent = S;
            S->Size++;
            if (S->Size == 1)
                S->First = t1;
            S->Last = t1;
            if (SS->Size == 0)
                SS->First = S;
            S->Parent = SS;
            SS->Last = S;
            if (S->Size == GroupSize) {
                S = S->Suc;
                SS->Size++;
                if (SS->Size == SGroupSize)
                    SS = SS->Suc;
            }
            t1->OldPredExcluded = t1->OldSucExcluded = 0;
            t1->Next = 0;
            if (KickType == 0 || Kicks == 0 || Trial == 1 ||
                !InBestTour(t1, t1->Pred) || !InBestTour(t1, t1->Suc))
                Activate(t1);
        } while ((t1 = t1->Suc) != FirstNode);
        if (S->Size < GroupSize)
            SS->Size++;
    }

    if (TSPTW_Makespan)
        CurrentCost = TSPTW_CurrentMakespanCost = TSPTW_MakespanCost();
    CurrentPenalty = INFINITY;
    CurrentPenalty = Penalty ? Penalty() : 0;
    if (TraceLevel >= 3 ||
        (TraceLevel == 2 &&
         (CurrentPenalty < BetterPenalty ||
          (CurrentPenalty == BetterPenalty && CurrentCost < BetterCost))))
        StatusReport(EntryTime, "");
    PredSucCostAvailable = 1;
    BIT_Update();

    /* Loop as long as improvements are found */
    while (true) {
        /* Choose t1 as the first "active" node */
        Node *t1;
        while ((t1 = RemoveFirstActive())) {
            /* t1 is now "passive" */
            // if ((TraceLevel >= 3 || (TraceLevel == 2 && Trial == 1)) &&
            //     ++it % (Dimension >= 100000 ? 10000 : Dimension >= 10000 ? 1000
            //                                                              : 100) ==
            //         0)
            //     printff("#%ld: Time = %0.6f sec.\n",
            //             it, fabs(GetTime() - EntryTime));
            /* Choose t2 as one of t1's two neighbors on the tour */
            for (Node *t2 : {PRED(t1), SUC(t1)}) {
                if (FixedOrCommon(t1, t2) ||
                    (RestrictedSearch && Near(t1, t2) &&
                     (Trial == 1 ||
                      (Trial > BackboneTrials &&
                       (KickType == 0 || Kicks == 0)))))
                    continue;
                GainType Gain = 0, G0 = C(t1, t2);
                /* Try to find a tour-improving chain of moves */
                do
                    t2 = Swaps == 0 ? BestMove(t1, t2, &G0, &Gain) : BestSubsequentMove(t1, t2, &G0, &Gain);
                while (t2);
                if (PenaltyGain > 0 || Gain > 0) {
                    /* An improvement has been found */
                    CurrentCost -= Gain;
                    CurrentPenalty -= PenaltyGain;
                    StoreTour();
                    TSPTW_CurrentMakespanCost = CurrentCost;
                    if (TraceLevel >= 3 ||
                        (TraceLevel == 2 &&
                         (CurrentPenalty < BetterPenalty ||
                          (CurrentPenalty == BetterPenalty &&
                           CurrentCost < BetterCost))))
                        StatusReport(EntryTime, "");
                    if (HashSearch(HTable, Hash, CurrentCost))
                        goto End_LinKernighan;
                    /* Make t1 "active" again */
                    Activate(t1);
                    OldSwaps = 0;
                    break;
                }
                OldSwaps = 0;
                RestoreTour();
                if (Asymmetric && SUC(t1) != t1)
                    Reversed ^= true;
            }
        }
        if (HashSearch(HTable, Hash, CurrentCost))
            goto End_LinKernighan;
        HashInsert(HTable, Hash, CurrentCost);
        /* Try to find improvements using non-sequential 4/5-opt moves */
        // auto cp = CurrentPenalty;
        // CurrentPenalty = INFINITY;
        // CurrentPenalty = Penalty ? Penalty() : 0;
        // ASSERT(cp == CurrentPenalty);
        if (Gain23Used) {
            PenaltyGain = 0;
            auto Gain = Gain23();
            if (Gain > 0 || PenaltyGain > 0) {
                /* An improvement has been found */
                CurrentCost -= Gain;
                CurrentPenalty -= PenaltyGain;
                TSPTW_CurrentMakespanCost = CurrentCost;
                StoreTour();
                if (TraceLevel >= 3 ||
                    (TraceLevel == 2 &&
                     (CurrentPenalty < BetterPenalty ||
                      (CurrentPenalty == BetterPenalty && CurrentCost < BetterCost))))
                    StatusReport(EntryTime, "+ ");
                if (HashSearch(HTable, Hash, CurrentCost))
                    goto End_LinKernighan;
                if (PenaltyGain > 0 || Gain > 0) {
                    continue;
                }
            }
        }
        break;
    }
End_LinKernighan:
    PredSucCostAvailable = 0;
    NormalizeNodeList();
    NormalizeSegmentList();
    Reversed = 0;
    return CurrentCost;
}
