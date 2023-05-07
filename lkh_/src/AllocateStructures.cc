#include "BIT.h"
#include "Heap.h"
#include "LKH.h"
#include "Segment.h"
#include "Sequence.h"

/*
 * The AllocateStructures function allocates all necessary
 * structures except nodes and candidates.
 */

#define Free(s)  \
    {            \
        free(s); \
        s = 0;   \
    }

void AllocateStructures() {
    int64_t i, K;

    Free(Heap);
    Free(BestTour);
    Free(BetterTour);
    Free(HTable);
    Free(Rand);
    Free(CacheSig);
    Free(CacheVal);
    Free(T);
    Free(G);
    Free(t);
    Free(p);
    Free(q);
    Free(SwapStack);
    Free(tSaved);

    HeapMake(Dimension);
    BestTour = new int64_t[1 + Dimension]();
    BetterTour = new int64_t[1 + Dimension]();
    HTable = new HashTable();
    HashInitialize((HashTable *)HTable);
    SRandom(Seed);
    Rand = new unsigned[Dimension + 1]();
    for (i = 1; i <= Dimension; i++)
        Rand[i] = Random();
    SRandom(Seed);
    if (WeightType != EXPLICIT) {
        for (i = 0; (1 << i) < (Dimension << 1); i++)
            ;
        i = 1 << i;
        CacheSig = new int64_t[i]();
        CacheVal = new int64_t[i]();
        CacheMask = i - 1;
    }
    AllocateSegments();
    K = MoveType;
    if (SubsequentMoveType > K)
        K = SubsequentMoveType;
    T = new Node *[1 + 2 * K]();
    G = new GainType[2 * K]();
    t = new Node *[6 * K]();
    tSaved = new Node *[1 + 2 * K]();
    p = new int64_t[6 * K]();
    q = new int64_t[6 * K]();
    incl = new int64_t[6 * K]();
    cycle = new int64_t[6 * K]();
    SwapStack = new SwapRecord[MaxSwaps + 6 * K]();
    BIT_Make(Dim_original);
}

/*
 * The AllocateSegments function allocates the segments of the two-level tree.
 */

void AllocateSegments() {
    Segment *S = 0, *SPrev;
    SSegment *SS = 0, *SSPrev;
    int64_t i;

    FreeSegments();
#ifdef THREE_LEVEL_TREE
    GroupSize = (int64_t)pow((double)Dimension, 1.0 / 3.0);
#elif defined TWO_LEVEL_TREE
    GroupSize = (int64_t)sqrt((double)Dimension);
#else
    GroupSize = Dimension;
#endif
    Groups = 0;
    for (i = Dimension, SPrev = 0; i > 0; i -= GroupSize, SPrev = S) {
        S = new Segment();
        S->Rank = ++Groups;
        if (!SPrev)
            FirstSegment = S;
        else
            Link(SPrev, S);
    }
    Link(S, FirstSegment);
#ifdef THREE_LEVEL_TREE
    SGroupSize = sqrt((double)Groups);
#else
    SGroupSize = Dimension;
#endif
    SGroups = 0;
    for (i = Groups, SSPrev = 0; i > 0; i -= SGroupSize, SSPrev = SS) {
        SS = new SSegment();
        SS->Rank = ++SGroups;
        if (!SSPrev)
            FirstSSegment = SS;
        else
            Link(SSPrev, SS);
    }
    Link(SS, FirstSSegment);
}
