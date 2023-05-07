#include "BIT.h"

#include "LKH.h"
#include "Segment.h"

/* An implementation of the binary indexed tree data structure (BIT)
 * proposed in
 *
 *      N. Mladenovic et al.,
 *      A general variable neighborhood search for the one-commodity
 *      pickup-and-delivery travelling salesman problem.
 *      European Journal of Operational Research, 220(1), 270–285, 2012.
 *
 * The BIT structure can be used to calculate the maximum (or minimum) of
 * an array of n elements in O(logn) time.
 *
 * The following operations are provided
 *
 *     BIT_make:         Create empty min and max trees.
 *     BIT_update:       Update the trees according to the current tour.
 *     BIT_LoadDiff3Opt,
 *     BIT_LoadDiff4Opt,
 *     BIT_LoadDiff5Opt,
 *     BIT_LoadDiff6Opt: Return the load difference for a proposed
 *                       3-, 4-, 5- or 6-opt move.
 */

static int64_t n, p;
static int64_t *MinTree;
static int64_t *MaxTree;
static int64_t *L;

#undef min
#undef max
static int64_t min(int64_t a, int64_t b) {
    return a < b ? a : b;
}

static int64_t max(int64_t a, int64_t b) {
    return a > b ? a : b;
}

static int64_t minSeq(int64_t a, int64_t b) {
    return abs(b - a) == n - 1 ? Dim_original : a < b ? a : b;
}

static int compare(const void *a, const void *b) {
    return *(int64_t *)a - *(int64_t *)b;
}

void BIT_Make(int64_t Size) {
    if (ProblemType != ONE_PDTSP)
        return;
    n = Size;
    for (p = 1; (1 << p) < n; p++)
        ;
    MinTree = new int64_t[1 << (p + 1)]();
    MaxTree = new int64_t[1 << (p + 1)]();
    L = new int64_t[n + 1]();
}

static void Build() {
    int64_t i, i1;
    for (i = 1; i <= n; i++) {
        i1 = i + (1 << p) - 1;
        MinTree[i1] = MaxTree[i1] = L[i];
    }
    for (i = (1 << p) - 1; i >= 1; i--) {
        MinTree[i] = min(MinTree[2 * i], MinTree[2 * i + 1]);
        MaxTree[i] = max(MaxTree[2 * i], MaxTree[2 * i + 1]);
    }
}

static int64_t BIT_Min(int64_t i, int64_t j) {
    int64_t vmin = INT_MAX;
    if (i > j)
        return vmin;
    i += (1 << p) - 2;
    j += (1 << p);
    for (; i / 2 != j / 2; i /= 2, j /= 2) {
        if ((i & 1) == 0)
            vmin = min(vmin, MinTree[i + 1]);
        if ((j & 1) != 0)
            vmin = min(vmin, MinTree[j - 1]);
    }
    return vmin;
}

static int64_t BIT_Max(int64_t i, int64_t j) {
    int64_t vmax = INT_MIN;
    if (i > j)
        return vmax;
    i += (1 << p) - 2;
    j += (1 << p);
    for (; i / 2 != j / 2; i /= 2, j /= 2) {
        if ((i & 1) == 0)
            vmax = max(vmax, MaxTree[i + 1]);
        if ((j & 1) != 0)
            vmax = max(vmax, MaxTree[j - 1]);
    }
    return vmax;
}

void BIT_Update() {
    if (ProblemType != ONE_PDTSP)
        return;
    bool Forward = SUC(Depot)->Id != Depot->Id + Dim_salesman;
    int64_t Load = 0, Seq = 0;
    Node *N = Depot;
    do {
        if (N->Id <= Dim_original) {
            N->Seq = ++Seq;
            L[Seq] = N->Load = Load += N->Demand;
            NodeSet[N->Id + Dim_salesman].Seq = Seq;
            NodeSet[N->Id + Dim_salesman].Load = Load;
        }
        N = Forward ? SUC(N) : PRED(N);
    } while (N != Depot);
    Build();
}

static int64_t LoadDiffKOpt(int64_t *t, int64_t K) {
    int64_t MinLoad = min(BIT_Min(1, t[0]), BIT_Min(t[2 * K - 1] + 1, Dim_original));
    int64_t MaxLoad = max(BIT_Max(1, t[0]), BIT_Max(t[2 * K - 1] + 1, Dim_original));
    int64_t Diff = 0, i, j;
    for (i = 0; i <= 2 * K - 4; i += 2) {
        Diff += L[t[i]] - L[t[i + 1]];
        j = t[i + 1] % Dim_original + 1;
        MinLoad = min(MinLoad, Diff + min(L[j], BIT_Min(j, t[i + 2])));
        MaxLoad = max(MaxLoad, Diff + max(L[j], BIT_Max(j, t[i + 2])));
    }
    return MaxLoad - MinLoad;
}

int64_t BIT_LoadDiff3Opt(Node *t1, Node *t2, Node *t3, Node *t4,
                         Node *t5, Node *t6) {
    if (ProblemType != ONE_PDTSP || Swaps > 0)
        return Capacity;
    int64_t s[3] = {minSeq(t1->Seq, t2->Seq),
                    minSeq(t3->Seq, t4->Seq),
                    minSeq(t5->Seq, t6->Seq)};
    qsort(s, 3, sizeof(int64_t), compare);
    int64_t t[6] = {s[0], s[1], s[2], s[0], s[1], s[2]};
    return LoadDiffKOpt(t, 3);
}

int64_t BIT_LoadDiff4Opt(Node *t1, Node *t2, Node *t3, Node *t4,
                         Node *t5, Node *t6, Node *t7, Node *t8) {
    if (ProblemType != ONE_PDTSP || Swaps > 0)
        return Capacity;
    int64_t s[4] = {minSeq(t1->Seq, t2->Seq),
                    minSeq(t3->Seq, t4->Seq),
                    minSeq(t5->Seq, t6->Seq),
                    minSeq(t7->Seq, t8->Seq)};
    qsort(s, 4, sizeof(int64_t), compare);
    int64_t t[8] = {s[0], s[2], s[3], s[1], s[2], s[0], s[1], s[3]};
    return LoadDiffKOpt(t, 4);
}

int64_t BIT_LoadDiff5Opt(Node *t1, Node *t2, Node *t3, Node *t4,
                         Node *t5, Node *t6, Node *t7, Node *t8,
                         Node *t9, Node *t10, int64_t Case10) {
    if (ProblemType != ONE_PDTSP || Swaps > 0)
        return Capacity;
    bool Forward = SUC(Depot)->Id != Depot->Id + Dim_salesman;
    int64_t s[5] = {minSeq(t1->Seq, t2->Seq),
                    minSeq(t3->Seq, t4->Seq),
                    minSeq(t5->Seq, t6->Seq),
                    minSeq(t7->Seq, t8->Seq),
                    minSeq(t9->Seq, t10->Seq)};
    qsort(s, 5, sizeof(int64_t), compare);
    if (Case10 == 4) {
        int64_t t[10] = {s[0], s[3], s[4], s[2], s[3],
                         s[1], s[2], s[0], s[1], s[4]};
        return LoadDiffKOpt(t, 5);
    }
    if (Case10 == 5) {
        if (BETWEEN(t6, Depot, t1)) {
            int64_t t[10] = {s[0], s[1], s[2], s[0], s[1],
                             s[3], s[4], s[2], s[3], s[4]};
            return LoadDiffKOpt(t, 5);
        }
        if (Forward ? BETWEEN(t8, Depot, t5) : BETWEEN(t6, Depot, t1)) {
            int64_t t[10] = {s[0], s[3], s[4], s[0], s[1],
                             s[2], s[3], s[1], s[2], s[4]};
            return LoadDiffKOpt(t, 5);
        }
        if (Forward ? BETWEEN(t4, Depot, t7) : BETWEEN(t10, Depot, t3)) {
            int64_t t[10] = {s[0], s[1], s[2], s[3], s[4],
                             s[2], s[3], s[0], s[1], s[4]};
            return LoadDiffKOpt(t, 5);
        }
        if (Forward ? BETWEEN(t10, Depot, t3) : BETWEEN(t4, Depot, t7)) {
            int64_t t[10] = {s[0], s[3], s[4], s[1], s[2],
                             s[0], s[1], s[2], s[3], s[4]};
            return LoadDiffKOpt(t, 5);
        }
        if (Forward || BETWEEN(t8, Depot, t5)) {
            int64_t t[10] = {s[0], s[2], s[3], s[1], s[2],
                             s[3], s[4], s[0], s[1], s[4]};
            return LoadDiffKOpt(t, 5);
        } else {
            int64_t t[10] = {s[0], s[3], s[4], s[0], s[1],
                             s[2], s[3], s[1], s[2], s[4]};
            return LoadDiffKOpt(t, 5);
        }
    }
    if (Case10 == 13) {
        if (Forward ? BETWEEN(t8, Depot, t1) : BETWEEN(t4, Depot, t7)) {
            int64_t t[10] = {s[0], s[1], s[2], s[3], s[4],
                             s[2], s[3], s[0], s[1], s[4]};
            return LoadDiffKOpt(t, 5);
        }
        if (Forward ? BETWEEN(t4, Depot, t7) : BETWEEN(t8, Depot, t1)) {
            int64_t t[10] = {s[0], s[3], s[4], s[1], s[2],
                             s[0], s[1], s[2], s[3], s[4]};
            return LoadDiffKOpt(t, 5);
        }
        if (Forward ? BETWEEN(t6, Depot, t3) : BETWEEN(t2, Depot, t9)) {
            int64_t t[10] = {s[0], s[2], s[3], s[1], s[2],
                             s[3], s[4], s[0], s[1], s[4]};
            return LoadDiffKOpt(t, 5);
        }
        if (BETWEEN(t10, Depot, t5)) {
            int64_t t[10] = {s[0], s[1], s[2], s[0], s[1],
                             s[3], s[4], s[2], s[3], s[4]};
            return LoadDiffKOpt(t, 5);
        } else {
            int64_t t[10] = {s[0], s[3], s[4], s[0], s[1],
                             s[2], s[3], s[1], s[2], s[4]};
            return LoadDiffKOpt(t, 5);
        }
    }
    if (Case10 == 14) {
        int64_t t[10] = {s[0], s[2], s[3], s[0], s[1],
                         s[3], s[4], s[1], s[2], s[4]};
        return LoadDiffKOpt(t, 5);
    }
    if (Case10 == 15) {
        if (Forward ? BETWEEN(t8, Depot, t1) : BETWEEN(t2, Depot, t5)) {
            int64_t t[10] = {s[0], s[3], s[4], s[1], s[2],
                             s[0], s[1], s[2], s[3], s[4]};
            return LoadDiffKOpt(t, 5);
        }
        if (Forward ? BETWEEN(t10, Depot, t7) : BETWEEN(t6, Depot, t3)) {
            int64_t t[10] = {s[0], s[2], s[3], s[1], s[2],
                             s[3], s[4], s[0], s[1], s[4]};
            return LoadDiffKOpt(t, 5);
        }
        if (BETWEEN(t4, Depot, t9)) {
            int64_t t[10] = {s[0], s[1], s[2], s[0], s[1],
                             s[3], s[4], s[2], s[3], s[4]};
            return LoadDiffKOpt(t, 5);
        } else if (Forward ? BETWEEN(t6, Depot, t3) : BETWEEN(t10, Depot, t7)) {
            int64_t t[10] = {s[0], s[3], s[4], s[0], s[1],
                             s[2], s[3], s[1], s[2], s[4]};
            return LoadDiffKOpt(t, 5);
        } else {
            int64_t t[10] = {s[0], s[1], s[2], s[3], s[4],
                             s[2], s[3], s[0], s[1], s[4]};
            return LoadDiffKOpt(t, 5);
        }
    }
    return 1;
}

int64_t BIT_LoadDiff6Opt(Node *t1, Node *t2, Node *t3, Node *t4,
                         Node *t5, Node *t6, Node *t7, Node *t8,
                         Node *t9, Node *t10, Node *t11, Node *t12) {
    if (ProblemType != ONE_PDTSP || Swaps > 0)
        return Capacity;
    int64_t s[6] = {minSeq(t1->Seq, t2->Seq),
                    minSeq(t3->Seq, t4->Seq),
                    minSeq(t5->Seq, t6->Seq),
                    minSeq(t7->Seq, t8->Seq),
                    minSeq(t9->Seq, t10->Seq),
                    minSeq(t11->Seq, t12->Seq)};
    qsort(s, 6, sizeof(int64_t), compare);
    int64_t t[12] = {s[0], s[4], s[5], s[3], s[4], s[2], s[3], s[1], s[2],
                     s[0], s[1], s[5]};
    int64_t r = LoadDiffKOpt(t, 6);
    return r;
}
