#include "LKH.h"

/*
 * The Distance_MTSP function computes the transformed distance for
 * an mTSP instance.
 */

int64_t Distance_MTSP(Node* Na, Node* Nb) {
    int64_t i;

    if (Fixed(Na, Nb))
        return 0;
    if (Forbidden(Na, Nb))
        return BIG_INT;
    if (Na->DepotId != 0 && Nb->DepotId != 0)
        return 0;
    if (Dim_salesman != Dimension) {
        if (Nb->DepotId && Na->Id <= Dim_original) {
            for (i = ExternalSalesmen; i >= 1; i--)
                if (Nb->DepotId == i)
                    break;
            if (i >= 1)
                return 0;
        }
        if (Na->DepotId && Nb->Id <= Dim_original) {
            for (i = ExternalSalesmen; i >= 1; i--)
                if (Na->DepotId == i)
                    break;
            if (i >= 1)
                return 0;
        }
        if (Na->DepotId != 0)
            Na = Na->Id <= Dim_salesman ? Depot : &NodeSet[Depot->Id + Dim_salesman];
        else if (Nb->DepotId != 0)
            Nb = Nb->Id <= Dim_salesman ? Depot : &NodeSet[Depot->Id + Dim_salesman];
    } else if (Dim_original != Dimension) {
        if (Na->Id > Dim_original)
            Na = Depot;
        if (Nb->Id > Dim_original)
            Nb = Depot;
    }
    return OldDistance(Na, Nb);
}
