#include "LKH.h"

/*
 * Functions for computing the transformed distance of an edge (Na,Nb).
 */

/*
 * The C_EXPLICIT function returns the distance by looking it up in a table.
 */

int64_t C_EXPLICIT(Node *Na, Node *Nb) {
    return Na->Id < Nb->Id ? Nb->C[Na->Id] : Na->C[Nb->Id];
}

/*
 * The C_FUNCTION function is used when the distance is defined by a
 * function (e.g. the Euclidean distance function). In order to speed
 * up the computations the following algorithm used:
 *
 *  (1) If (Na,Nb) is an edge on the current tour, then its distance
 *      is available in either the field PredCost or SucCost.
 *
 *  (2) If the edge (Na,Nb) is a candidate edge incident to Na, then
 *      its distance is available in the field Cost of the corresponding
 *      Candidate structure.
 *
 *  (3) A hash table (CacheVal) is consulted to see if the distance has
 *      been stored.
 *
 *  (4) Otherwise the distance function is called and the distance computed
 *      is stored in the hash table.
 */

int64_t C_FUNCTION(Node *Na, Node *Nb) {
    Node *Nc;
    Candidate *Cand;
    int64_t Index, i, j;

    if (CostMatrix)
        return D(Na, Nb);
    if (PredSucCostAvailable) {
        if (Na->Suc == Nb)
            return Na->SucCost;
        if (Na->Pred == Nb)
            return Na->PredCost;
    }
    if ((Cand = Na->CandidateSet.data()))
        for (; (Nc = Cand->To); Cand++)
            if (Nc == Nb)
                return Cand->Cost;
    if ((Cand = Nb->CandidateSet.data()))
        for (; (Nc = Cand->To); Cand++)
            if (Nc == Na)
                return Cand->Cost;
    if ((Cand = Na->BackboneCandidateSet))
        for (; (Nc = Cand->To); Cand++)
            if (Nc == Nb)
                return Cand->Cost;
    if ((Cand = Nb->BackboneCandidateSet))
        for (; (Nc = Cand->To); Cand++)
            if (Nc == Na)
                return Cand->Cost;
    if (CacheSig == 0)
        return D(Na, Nb);
    i = Na->Id;
    j = Nb->Id;
    if (i > j) {
        int64_t k = i;
        i = j;
        j = k;
    }
    Index = ((i << 8) + i + j) & CacheMask;
    if (CacheSig[Index] == i)
        return CacheVal[Index];
    CacheSig[Index] = i;
    return (CacheVal[Index] = D(Na, Nb));
}

int64_t D_EXPLICIT(Node *Na, Node *Nb) {
    return (Na->Id <
                    Nb->Id
                ? Nb->C[Na->Id]
                : Na->C[Nb->Id]) +
           Na->Pi + Nb->Pi;
}

int64_t D_FUNCTION(Node *Na, Node *Nb) {
    return (Fixed(Na, Nb) ? 0 : Distance(Na, Nb)) + Na->Pi + Nb->Pi;
}

/* Functions for computing lower bounds for the distance functions */

int64_t c_ATT(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t dx = (int64_t)(ceil(Scale * 0.31622 * fabs(Na->X - Nb->X))),
        dy = (int64_t)(ceil(Scale * 0.31622 * fabs(Na->Y - Nb->Y)));
    return (dx > dy ? dx : dy) + Na->Pi + Nb->Pi;
}

int64_t c_CEIL_2D(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t dx = (int64_t)ceil(Scale * fabs(Na->X - Nb->X)),
        dy = (int64_t)ceil(Scale * fabs(Na->Y - Nb->Y));
    return (dx > dy ? dx : dy) + Na->Pi + Nb->Pi;
}

int64_t c_CEIL_3D(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t dx = (int64_t)ceil(Scale * fabs(Na->X - Nb->X)),
        dy = (int64_t)ceil(Scale * fabs(Na->Y - Nb->Y)),
        dz = (int64_t)ceil(Scale * fabs(Na->Z - Nb->Z));
    if (dy > dx)
        dx = dy;
    if (dz > dx)
        dx = dz;
    return dx + Na->Pi + Nb->Pi;
}

int64_t c_EUC_2D(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t dx = (int64_t)(Scale * fabs(Na->X - Nb->X) + 0.5),
        dy = (int64_t)(Scale * fabs(Na->Y - Nb->Y) + 0.5);
    return (dx > dy ? dx : dy) + Na->Pi + Nb->Pi;
}

int64_t c_EUC_3D(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t dx = (int64_t)(Scale * fabs(Na->X - Nb->X) + 0.5),
        dy = (int64_t)(Scale * fabs(Na->Y - Nb->Y) + 0.5),
        dz = (int64_t)(Scale * fabs(Na->Z - Nb->Z) + 0.5);
    if (dy > dx)
        dx = dy;
    if (dz > dx)
        dx = dz;
    return dx + Na->Pi + Nb->Pi;
}

int64_t c_FLOOR_2D(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t dx = (int64_t)floor(Scale * fabs(Na->X - Nb->X)),
        dy = (int64_t)floor(Scale * fabs(Na->Y - Nb->Y));
    return (dx > dy ? dx : dy) + Na->Pi + Nb->Pi;
}

int64_t c_FLOOR_3D(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t dx = (int64_t)floor(Scale * fabs(Na->X - Nb->X)),
        dy = (int64_t)floor(Scale * fabs(Na->Y - Nb->Y)),
        dz = (int64_t)floor(Scale * fabs(Na->Z - Nb->Z));
    if (dy > dx)
        dx = dy;
    if (dz > dx)
        dx = dz;
    return dx + Na->Pi + Nb->Pi;
}

#define PI 3.141592
#define RRR 6378.388

int64_t c_GEO(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t da = (int64_t)Na->X, db = (int64_t)Nb->X;
    double ma = Na->X - da, mb = Nb->X - db;
    int64_t dx =
        (int64_t)Scale * (RRR * PI / 180.0 * fabs(da - db + 5.0 * (ma - mb) / 3.0) + 1.0);
    return dx + Na->Pi + Nb->Pi;
}

#undef M_PI
#define M_PI 3.14159265358979323846264
#define M_RRR 6378388.0

int64_t c_GEOM(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t dx = (int64_t)Scale * (M_RRR * M_PI / 180.0 * fabs(Na->X - Nb->X) + 1.0);
    return dx + Na->Pi + Nb->Pi;
}

#define f (1 - 1 / 298.257)

int64_t c_GEO_MEEUS(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t da = (int64_t)Na->X, db = (int64_t)Nb->X;
    double ma = Na->X - da, mb = Nb->X - db;
    int64_t dx = (int64_t)(Scale * RRR * M_PI / 180.0 * fabs(da - db + 5.0 * (ma - mb) / 3.0) * f + 0.5);
    return dx + Na->Pi + Nb->Pi;
}

int64_t c_GEOM_MEEUS(Node *Na, Node *Nb) {
    if (Fixed(Na, Nb))
        return Na->Pi + Nb->Pi;
    int64_t dx = (int64_t)(Scale * M_RRR * M_PI / 180.0 * fabs(Na->X - Nb->X) *
                       f +
                   0.5);
    return dx + Na->Pi + Nb->Pi;
}
