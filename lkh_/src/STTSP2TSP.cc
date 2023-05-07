#include "Heap.h"
#include "LKH.h"

static void Dijkstra(Node *Source);

void STTSP2TSP() {
    int64_t NewDimension = 0, i, j, k;
    int64_t **Matrix;
    Node *N1 = FirstNode, *N2;

    i = 0;
    do {
        if (N1->Required) {
            NewDimension++;
            N1->Serial = i++;
        }
    } while ((N1 = N1->Suc) != FirstNode);
    Matrix = new int64_t *[NewDimension]();
    for (i = 0; i < NewDimension; i++)
        Matrix[i] = new int64_t[NewDimension]();
    do {
        if (N1->Required) {
            Dijkstra(N1);
            N1->PathLength = new int64_t[NewDimension + 1]();
            N1->Path = new int64_t *[NewDimension + 1]();
            i = N1->Serial;
            N2 = FirstNode;
            do {
                if (N2 != N1 && N2->Required) {
                    j = N2->Serial;
                    Matrix[i][j] = N2->Cost;
                    Node *N = N2;
                    while ((N = N->Dad) != N1)
                        N1->PathLength[j + 1]++;
                    if (N1->PathLength[j + 1] > 0) {
                        N1->Path[j + 1] =
                            new int64_t[N1->PathLength[j + 1]]();
                        k = N1->PathLength[j + 1];
                        N = N2;
                        while ((N = N->Dad) != N1)
                            N1->Path[j + 1][--k] = N->OriginalId;
                    }
                }
            } while ((N2 = N2->Suc) != FirstNode);
        }
    } while ((N1 = N1->Suc) != FirstNode);
    j = 0;
    for (i = 1; i <= Dimension; i++) {
        N1 = &NodeSet[i];
        if (N1->Required) {
            N1->Id = N1->Serial + 1;
            N1->C = Matrix[N1->Serial] - 1;
            N1->CandidateSet.clear();
            NodeSet[++j] = *N1;
        }
    }
    for (i = 1; i <= NewDimension; i++, N1 = N2) {
        N2 = &NodeSet[i];
        if (i == 1)
            FirstNode = N2;
        else
            Link(N1, N2);
    }
    Link(N1, FirstNode);
    Dimension = Dim_salesman = NewDimension;
    WeightType = EXPLICIT;
    Distance = Distance_EXPLICIT;
}

static void Dijkstra(Node *Source) {
    Node *Blue;
    Node *N;
    int64_t d;

    Blue = N = Source;
    Blue->Dad = 0;
    Blue->Loc = 0;
    Blue->Cost = 0;
    HeapClear();
    while ((N = N->Suc) != Source) {
        N->Dad = Blue;
        N->Cost = N->Rank = INT_MAX / 2;
        HeapLazyInsert(N);
    }
    for (auto *NBlue = Blue->CandidateSet.data(); (N = NBlue->To); NBlue++) {
        N->Dad = Blue;
        N->Cost = N->Rank = NBlue->Cost;
        HeapSiftUp(N);
    }
    while ((Blue = HeapDeleteMin())) {
        for (auto *NBlue = Blue->CandidateSet.data(); (N = NBlue->To); NBlue++) {
            if (N->Loc && (d = Blue->Cost + NBlue->Cost) < N->Cost) {
                N->Dad = Blue;
                N->Cost = N->Rank = d;
                HeapSiftUp(N);
            }
        }
    }
}