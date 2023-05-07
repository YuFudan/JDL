#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

#include "Genetic.h"
#include "Heap.h"
#include "LKH.h"
#include "Segment.h"

namespace py = pybind11;
using namespace py::literals;

void init_parameters(int64_t seed, int64_t trace_level) {
    ProblemFileName = PiFileName = InputTourFileName = OutputTourFileName = TourFileName = 0;
    CandidateFiles = MergeTourFiles = 0;
    AscentCandidates = 50;
    BackboneTrials = 0;
    Backtracking = 0;
    BWTSP_B = 0;
    BWTSP_Q = 0;
    BWTSP_L = INT_MAX;
    CandidateSetSymmetric = 0;
    CandidateSetType = ALPHA;
    Crossover = ERXT;
    DelaunayPartitioning = 0;
    DelaunayPure = 0;
    DemandDimension = 1;
    DistanceLimit = INFINITY;
    Excess = -1;
    ExternalSalesmen = 0;
    ExtraCandidates = 0;
    ExtraCandidateSetSymmetric = 0;
    ExtraCandidateSetType = QUADRANT;
    Gain23Used = 1;
    GainCriterionUsed = 1;
    GridSize = 1000000.0;
    InitialPeriod = -1;
    InitialStepSize = 1;
    InitialTourAlgorithm = WALK;
    InitialTourFraction = 1.0;
    KarpPartitioning = 0;
    KCenterPartitioning = 0;
    KMeansPartitioning = 0;
    Kicks = 1;
    KickType = 0;
    MaxBreadth = INT_MAX;
    MaxCandidates = 5;
    MaxPopulationSize = 0;
    MaxSwaps = -1;
    MaxTrials = -1;
    MoorePartitioning = 0;
    MoveType = 5;
    MoveTypeSpecial = 0;
    MTSPDepot = 1;
    MTSPMinSize = 1;
    MTSPMaxSize = -1;
    MTSPObjective = -1;
    NonsequentialMoveType = -1;
    Optimum = -INFINITY;
    PatchingA = 1;
    PatchingC = 0;
    PatchingAExtended = 0;
    PatchingARestricted = 0;
    PatchingCExtended = 0;
    PatchingCRestricted = 0;
    POPMUSIC_InitialTour = 0;
    POPMUSIC_MaxNeighbors = 5;
    POPMUSIC_SampleSize = 10;
    POPMUSIC_Solutions = 50;
    POPMUSIC_Trials = 1;
    Recombination = IPT;
    RestrictedSearch = 1;
    RohePartitioning = 0;
    Runs = 10;
    Salesmen = 1;
    Scale = 1;
    Seed = 1;
    SierpinskiPartitioning = 0;
    StopAtOptimum = 1;
    Subgradient = 1;
    SubproblemBorders = 0;
    SubproblemsCompressed = 0;
    SubproblemSize = 0;
    SubsequentMoveType = 0;
    SubsequentMoveTypeSpecial = 0;
    SubsequentPatching = 1;
    TimeLimit = INFINITY;
    TraceLevel = 1;
    TSPTW_Makespan = 0;

    // SPECIAL
    Gain23Used = 0;
    KickType = 4;
    MaxSwaps = 0;
    MoveType = 5;
    MoveTypeSpecial = 1;
    MaxPopulationSize = 10;

    MaxTrials = 10;
    Seed = seed;
    TraceLevel = trace_level;
    // OutputTourFileName = xxx;
    // PiFileName = xxx;
    // ProblemFileName = xxx;
    // Runs = xxx;
    // Seed = xxx;
    // PopulationSize = xxx;
    // TraceLevel = xxx;

    StartTime = GetTime();
    MaxMatrixDimension = 20000;
    MergeWithTour = MergeWithTourIPT;
    CostTimeMatrix = NULL;
    CostDistanceMatrix = NULL;
    TruePenalty = -1;
}

void init_problem(const char *type, int64_t dimension, int64_t num_vehicle, int64_t capacity,
                  py::array_t<double, py::array::c_style | py::array::forcecast> dist_mat, py::array_t<double, py::array::c_style | py::array::forcecast> time_mat, py::array_t<double, py::array::c_style | py::array::forcecast> demand_mat) {
    if (FirstNode) {
        delete CostMatrix;
        delete CostDistanceMatrix;
        delete CostTimeMatrix;
        NodeSet.clear();
    }
    FirstNode = nullptr;
    WeightType = WeightFormat = ProblemType = -1;
    CoordType = NO_COORDS;
    Name = nullptr;
    EdgeDataFormat = NodeCoordType = DisplayDataType = nullptr;
    Distance = nullptr;
    C = nullptr;
    c = nullptr;

    if (strcmp(type, "VRPSPDTW") == 0) {
        ProblemType = VRPSPDTW;
    } else {
        eprintf("Unknown problem type");
    }
    Asymmetric =
        ProblemType == ATSP ||
        ProblemType == CCVRP ||
        ProblemType == ACVRP ||
        ProblemType == ADCVRP ||
        ProblemType == CVRPTW ||
        ProblemType == MLP ||
        ProblemType == M_PDTSP ||
        ProblemType == M1_PDTSP ||
        ProblemType == ONE_PDTSP ||
        ProblemType == OVRP ||
        ProblemType == PDTSP ||
        ProblemType == PDTSPF ||
        ProblemType == PDTSPL ||
        ProblemType == PDPTW ||
        ProblemType == RCTVRP ||
        ProblemType == RCTVRPTW ||
        ProblemType == SOP ||
        ProblemType == TRP ||
        ProblemType == TSPDL ||
        ProblemType == TSPTW ||
        ProblemType == VRPB ||
        ProblemType == VRPBTW || ProblemType == VRPSPDTW;
    Dim_original = dimension;
    WeightType = EXPLICIT;
    Distance = Distance_EXPLICIT;
    WeightFormat = FULL_MATRIX;
    Salesmen = num_vehicle;
    Capacity = capacity;
    {
        CreateNodes();
        ASSERT(Asymmetric);
        CostMatrix = new int64_t[Dim_salesman * Dim_salesman];
        for (int64_t i = 0; i < Dim_salesman; ++i) {
            NodeSet[i + 1].C = CostMatrix + (i * Dim_salesman - 1);
        }

        if (ProblemType == HPP)
            Dimension--;
        if (WeightFormat != FULL_MATRIX) {
            NOT_IMP
        }
        auto *pos = CostDistanceMatrix = new double[Dim_original * Dim_original];
        for (int64_t i = 1; i <= Dim_original; ++i) {
            NodeSet[i].CD = pos;
            pos += Dim_original;
        }
        {
            auto r = dist_mat.unchecked<2>();
            ASSERT(r.shape(0) == Dim_original);
            ASSERT(r.shape(1) == Dim_original);
            auto *p = CostDistanceMatrix;
            for (py::ssize_t i = 0; i < Dim_original; ++i) {
                auto &Ni = NodeSet[i + 1];
                for (py::ssize_t j = 0; j < Dim_original; ++j) {
                    Ni.C[j + 1] = round(*p++ = r(i, j));
                }
            }
        }
        for (int64_t i = Dim_original + 1; i <= Dimension; ++i) {
            NodeSet[i].CD = NodeSet[NodeSet[i].Index + 1].CD;
        }
        if (ProblemType == HPP)
            Dimension++;
        if (Asymmetric) {
            for (int64_t i = 1; i <= Dim_salesman; ++i)
                FixEdge(&NodeSet[i], &NodeSet[i + Dim_salesman]);
            Distance = Distance_ATSP;
            WeightType = -1;
        }
    }
    {
        auto *pos = CostTimeMatrix = new double[Dim_original * Dim_original];
        for (int64_t i = 1; i <= Dim_original; ++i) {
            NodeSet[i].CT = pos;
            pos += Dim_original;
        }
        for (int64_t i = Dim_original + 1; i <= Dimension; ++i) {
            NodeSet[i].CT = NodeSet[NodeSet[i].Index + 1].CT;
        }
        {
            auto r = time_mat.unchecked<2>();
            ASSERT(r.shape(0) == Dim_original);
            ASSERT(r.shape(1) == Dim_original);
            auto *p = CostTimeMatrix;
            for (py::ssize_t i = 0; i < Dim_original; ++i) {
                for (py::ssize_t j = 0; j < Dim_original; ++j) {
                    *p++ = r(i, j);
                }
            }
        }
    }

    {
        auto r = demand_mat.unchecked<2>();
        ASSERT(r.shape(0) == Dim_original);
        ASSERT(r.shape(1) == 5);
        auto *p = &NodeSet[1];
        for (int64_t i = 0; i < Dim_original; ++i) {
            p->Demand = 0;
            p->Pickup = r(i, 0);
            p->Delivery = r(i, 1);
            p->Earliest = r(i, 2);
            p->Latest = r(i, 3);
            p->ServiceTime = r(i, 4);
            ++p;
        }
        if (ProblemType != VRPSPDTW) {
            for (int64_t i = 1; i <= Dim_original; ++i) {
                auto &N = NodeSet[i];
                if (N.Delivery) {
                    if (NodeSet[N.Delivery].Pickup != N.Id ||
                        N.Delivery == N.Id)
                        eprintf(
                            "PICKUP_AND_DELIVERY_SECTION: "
                            "Illegal pairing for node %ld",
                            N.Id);
                    if (N.Demand < 0)
                        eprintf(
                            "PICKUP_AND_DELIVERY_SECTION: "
                            "Negative demand for delivery node %ld",
                            N.Id);
                } else if (N.Pickup) {
                    if (NodeSet[N.Pickup].Delivery != N.Id || N.Pickup == N.Id)
                        eprintf(
                            "PICKUP_AND_DELIVERY_SECTION: "
                            "Illegal pairing for node %ld",
                            N.Id);
                    if (N.Demand > 0)
                        eprintf(
                            "PICKUP_AND_DELIVERY_SECTION: "
                            "Positive demand for pickup node %ld",
                            N.Id);
                    if (N.Demand + NodeSet[N.Pickup].Demand)
                        eprintf(
                            "PICKUP_AND_DELIVERY_SECTION: "
                            "Demand for pickup node %ld and demand for delivery "
                            "node %ld does not sum to zero",
                            N.Id,
                            N.Pickup);
                }
            }
        }
    }

    MTSPDepot = 1;
    Swaps = 0;

    if (Seed == 0)
        Seed = (unsigned)time(0);
    if (MaxSwaps < 0)
        MaxSwaps = Dimension;
    if (KickType > Dimension / 2)
        KickType = Dimension / 2;
    if (MaxCandidates > Dimension - 1)
        MaxCandidates = Dimension - 1;
    if (ExtraCandidates > Dimension - 1)
        ExtraCandidates = Dimension - 1;
    if (SubproblemSize >= Dimension)
        SubproblemSize = Dimension;
    else if (SubproblemSize == 0) {
        if (AscentCandidates > Dimension - 1)
            AscentCandidates = Dimension - 1;
        if (InitialPeriod < 0) {
            InitialPeriod = Dimension / 2;
            if (InitialPeriod < 100)
                InitialPeriod = 100;
        }
        if (Excess < 0)
            Excess = 1.0 / Dim_salesman * Salesmen;
        if (MaxTrials == -1)
            MaxTrials = Dimension;
        HeapMake(Dimension);
    }
    if (POPMUSIC_MaxNeighbors > Dimension - 1)
        POPMUSIC_MaxNeighbors = Dimension - 1;
    if (POPMUSIC_SampleSize > Dimension)
        POPMUSIC_SampleSize = Dimension;
    Depot = &NodeSet[MTSPDepot];
    if (ProblemType == CVRP) {
        Node *N;
        int64_t MinSalesmen;
        if (Capacity <= 0)
            eprintf("CAPACITY not specified");
        TotalDemand = 0;
        N = FirstNode;
        do
            TotalDemand += N->Demand;
        while ((N = N->Suc) != FirstNode);
        MinSalesmen =
            TotalDemand / Capacity + (TotalDemand % Capacity != 0);
        if (Salesmen == 1) {
            Salesmen = MinSalesmen;
            if (Salesmen > Dimension)
                eprintf("CVRP: SALESMEN larger than DIMENSION");
        } else if (Salesmen < MinSalesmen)
            eprintf("CVRP: SALESMEN too small to meet demand");
        ASSERT(Salesmen >= 1 && Salesmen <= Dimension);
        if (Salesmen == 1)
            ProblemType = TSP;
        Penalty = Penalty_CVRP;
    }
    if (ProblemType == TSPTW) {
        Salesmen = 1;
        Penalty = Penalty_TSPTW;
    } else
        TSPTW_Makespan = 0;
    if (Salesmen > 1) {
        if (Salesmen > Dim_original && MTSPMinSize > 0)
            eprintf("Too many salesmen/vehicles (>= DIMENSION)");
        MTSP2TSP();
    }
    if (ProblemType == STTSP)
        STTSP2TSP();
    if (ProblemType == ACVRP || ProblemType == ADCVRP)
        Penalty = Penalty_ACVRP;
    else if (ProblemType == CCVRP)
        Penalty = Penalty_CCVRP;
    else if (ProblemType == CTSP)
        Penalty = Penalty_CTSP;
    else if (ProblemType == CVRPTW)
        Penalty = Penalty_CVRPTW;
    else if (ProblemType == MLP)
        Penalty = Penalty_MLP;
    else if (ProblemType == OVRP)
        Penalty = Penalty_OVRP;
    else if (ProblemType == PDTSP)
        Penalty = Penalty_PDTSP;
    else if (ProblemType == PDTSPF)
        Penalty = Penalty_PDTSPF;
    else if (ProblemType == PDTSPL)
        Penalty = Penalty_PDTSPL;
    else if (ProblemType == PDPTW)
        Penalty = Penalty_PDPTW;
    else if (ProblemType == ONE_PDTSP)
        Penalty = Penalty_1_PDTSP;
    else if (ProblemType == M_PDTSP)
        Penalty = Penalty_M_PDTSP;
    else if (ProblemType == M1_PDTSP)
        Penalty = Penalty_M1_PDTSP;
    else if (ProblemType == RCTVRP || ProblemType == RCTVRPTW)
        Penalty = Penalty_RCTVRP;
    else if (ProblemType == TRP)
        Penalty = Penalty_TRP;
    else if (ProblemType == TRPP)
        Penalty = Penalty_TRPP;
    else if (ProblemType == TSPDL)
        Penalty = Penalty_TSPDL;
    else if (ProblemType == TSPPD)
        Penalty = Penalty_TSPPD;
    if (ProblemType == VRPB)
        Penalty = Penalty_VRPB;
    else if (ProblemType == VRPBTW)
        Penalty = Penalty_VRPBTW;
    else if (ProblemType == VRPSPDTW)
        Penalty = Penalty_VRPSPDTW;
    if (Penalty && (SubproblemSize > 0 || SubproblemTourFile))
        eprintf("Partitioning not implemented for constrained problems");
    Depot->DepotId = 1;
    for (int64_t i = Dim_original + 1; i <= Dim_salesman; ++i)
        NodeSet[i].DepotId = i - Dim_original + 1;
    if (Dimension != Dim_salesman) {
        NodeSet[Depot->Id + Dim_salesman].DepotId = 1;
        for (int64_t i = Dim_original + 1; i <= Dim_salesman; ++i)
            NodeSet[i + Dim_salesman].DepotId = i - Dim_original + 1;
    }
    ASSERT(Scale == 1);
    ASSERT(ServiceTime == 0);
    ASSERT(CostMatrix);
    if (ProblemType == TSPTW ||
        ProblemType == CVRPTW || ProblemType == VRPBTW ||
        ProblemType == PDPTW || ProblemType == RCTVRPTW) {
        for (int64_t i = 1; i <= Dim_original; ++i) {
            Node *Ni = &NodeSet[i];
            for (int64_t j = 1; j <= Dim_original; j++) {
                Node *Nj = &NodeSet[j];
                if (Ni != Nj &&
                    Ni->Earliest + Ni->ServiceTime + Ni->C[j] > Nj->Latest)
                    Ni->C[j] = BIG_INT;
            }
        }
        if (ProblemType == TSPTW) {
            for (int64_t i = 1; i <= Dim_original; ++i)
                for (int64_t j = 1; j <= Dim_original; j++)
                    if (j != i)
                        NodeSet[i].C[j] += NodeSet[i].ServiceTime;
        }
    }
    C = WeightType == EXPLICIT ? C_EXPLICIT : C_FUNCTION;
    D = WeightType == EXPLICIT ? D_EXPLICIT : D_FUNCTION;
    if (ProblemType != CVRP && ProblemType != CVRPTW &&
        ProblemType != CTSP && ProblemType != STTSP &&
        ProblemType != TSP && ProblemType != ATSP) {
        for (int64_t i = Dim_original + 1; i <= Dim_salesman; ++i) {
            for (int64_t j = 1; j <= Dim_salesman; j++) {
                if (j == i)
                    continue;
                if (j == MTSPDepot || j > Dim_original)
                    NodeSet[i].C[j] = NodeSet[MTSPDepot].C[j] = BIG_INT;
                NodeSet[i].C[j] = NodeSet[MTSPDepot].C[j];
                NodeSet[j].C[i] = NodeSet[j].C[MTSPDepot];
            }
        }
        if (ProblemType == CCVRP || ProblemType == OVRP)
            for (int64_t i = 1; i <= Dim_original; ++i)
                NodeSet[i].C[MTSPDepot] = 0;
    }
    if (SubsequentMoveType == 0) {
        SubsequentMoveType = MoveType;
        SubsequentMoveTypeSpecial = MoveTypeSpecial;
    }
    int64_t K = MoveType >= SubsequentMoveType || !SubsequentPatching ? MoveType : SubsequentMoveType;
    if (PatchingC > K)
        PatchingC = K;
    if (PatchingA > 1 && PatchingA >= PatchingC)
        PatchingA = PatchingC > 2 ? PatchingC - 1 : 1;
    if (NonsequentialMoveType == -1 ||
        NonsequentialMoveType > K + PatchingC + PatchingA - 1)
        NonsequentialMoveType = K + PatchingC + PatchingA - 1;
    if (PatchingC >= 1) {
        BestMove = BestSubsequentMove = BestKOptMove;
        if (!SubsequentPatching && SubsequentMoveType <= 5) {
            MoveFunction BestOptMove[] =
                {0, 0, Best2OptMove, Best3OptMove,
                 Best4OptMove, Best5OptMove};
            BestSubsequentMove = BestOptMove[SubsequentMoveType];
        }
    } else {
        MoveFunction BestOptMove[] = {0, 0, Best2OptMove, Best3OptMove,
                                      Best4OptMove, Best5OptMove};
        BestMove = MoveType <= 5 ? BestOptMove[MoveType] : BestKOptMove;
        BestSubsequentMove = SubsequentMoveType <= 5 ? BestOptMove[SubsequentMoveType] : BestKOptMove;
    }
    if (MoveTypeSpecial)
        BestMove = BestSpecialOptMove;
    if (SubsequentMoveTypeSpecial)
        BestSubsequentMove = BestSpecialOptMove;
    if (ProblemType == HCP || ProblemType == HPP)
        MaxCandidates = 0;
    if (TraceLevel >= 1) {
        printff("done\n");
        PrintParameters();
    }
}

py::array_t<double> get_distance_matrix() {
    ASSERT(CostDistanceMatrix);
    return py::array_t<double>({Dim_original, Dim_original}, CostDistanceMatrix);
}

py::array_t<double> get_time_matrix() {
    ASSERT(CostTimeMatrix);
    return py::array_t<double>({Dim_original, Dim_original}, CostTimeMatrix);
}

void preprocess() {
    AllocateStructures();
    if (ProblemType == TSPTW)
        TSPTW_Reduce();
    if (ProblemType == VRPB || ProblemType == VRPBTW)
        VRPB_Reduce();
    if (ProblemType == PDPTW)
        PDPTW_Reduce();
    CreateCandidateSet();
    InitializeStatistics();
    ASSERT(Norm != 0 || Penalty);
    Norm = 9999;
    BestCost = INFINITY;
    BestPenalty = CurrentPenalty = INFINITY;
    HashInitialize(HTable);
    auto t = FirstNode;
    do
        t->OldPred = t->OldSuc = t->NextBestSuc = t->BestSuc = 0;
    while ((t = t->Suc) != FirstNode);
}

void set_trial(int64_t trial) {
    Trial = trial;
}

int64_t get_trial() {
    return Trial;
}

void generate_solution() {
    static std::vector<int64_t> ret;
    ASSERT(Trial == 1 || FirstNode->BestSuc);
    FirstNode = &NodeSet[1 + Random() % Dimension];
    ChooseInitialTour();
}

py::array_t<int64_t> get_solution() {
    static std::vector<int64_t> ret;
    ret.resize(Dim_salesman);
    auto *t = &NodeSet[1];
    for (auto &i : ret) {
        i = t->Id - 1;
        t = t->Suc;
        ASSERT(t->Id = i + 1 + Dim_salesman);
        t = t->Suc;
    }
    ASSERT(t == &NodeSet[1]);
    return py::array_t<int64_t>(Dim_salesman, ret.data());
}

int64_t id2rl(int64_t i) {
    return i >= Dim_original ? i - Dim_original : i + Salesmen - 1;
}

int64_t rl2id(int64_t i) {
    return i < Salesmen - 1 ? i + Dim_original : i - Salesmen + 1;
}

void set_rl_solution(py::array_t<int64_t> arr) {
    auto r = arr.unchecked<1>();
    ASSERT(r.shape(0) == Dim_salesman);
    for (py::ssize_t i = 0; i < Dim_salesman; ++i) {
        Link(&NodeSet[1 + rl2id(i) + Dim_salesman], &NodeSet[1 + rl2id(r(i))]);
    }
}

void set_dummy_edge() {
    for (int64_t i = 1; i <= Dim_salesman; ++i) {
        Link(&NodeSet[i], &NodeSet[i + Dim_salesman]);
    }
}

py::array_t<int64_t> get_rl_solution() {
    // 新加的仓库编号为0~Salesmen-2，原来的仓库变为Salesmen-1
    static std::vector<int64_t> ret;
    ret.resize(Dim_salesman);
    auto *t = &NodeSet[1];
    int64_t last = Salesmen - 1;
    ASSERT(t->Id == 1);
    for (int64_t i = 0; i < Dim_salesman; ++i) {
        ASSERT(t->Suc->Id = t->Id + Dim_salesman);
        t = t->Suc->Suc;
        last = ret[last] = id2rl(t->Id - 1);
    }
    ASSERT(t == &NodeSet[1]);
    return py::array_t<int64_t>(Dim_salesman, ret.data());
}

void lkh_begin() {
    Reversed = false;
    FirstActive = LastActive = nullptr;
    Swaps = 0;
    CurrentCost = 0;
    Hash = 0;

    // 初始化segment结构
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

    // 计算初始回路的Cost和Hash
    // 标记active结点
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

    CurrentPenalty = INFINITY;
    CurrentPenalty = Penalty ? Penalty() : 0;
    PredSucCostAvailable = 1;
}

py::array_t<int64_t> lkh_active_positions() {
    static std::vector<int64_t> ret;
    ret.clear();
    ASSERT(FirstActive);
    auto *t = FirstActive;
    do {
        ret.push_back(t->Id - 1);
        t = t->Next;
    } while (t != LastActive);
    return py::array_t<int64_t>(ret.size(), ret.data());
}

int64_t lkh_pop_active_pos() {
    auto *t = RemoveFirstActive();
    return t ? t->Id - 1 : -1;
}

void lkh_activate(int64_t pos) {
    Activate(&NodeSet[pos + 1]);
}

// 尝试在pos位置作用lkh，返回penalty和cost的下降
std::tuple<double, double> lkh_try(int64_t pos) {
    Node *t1 = &NodeSet[pos + 1];
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
            return {PenaltyGain, Gain};
        }
        OldSwaps = 0;
        RestoreTour();
        if (Asymmetric && SUC(t1) != t1)
            Reversed ^= true;
    }
    return {0, 0};
}

// 在pos位置作用lkh，返回是否找到更好的解
bool lkh_apply(int64_t pos) {
    Node *t1 = &NodeSet[pos + 1];
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
            // if (HashSearch(HTable, Hash, CurrentCost))
            //     goto End_LinKernighan;
            OldSwaps = 0;
            return true;
        }
        OldSwaps = 0;
        RestoreTour();
        if (Asymmetric && SUC(t1) != t1)
            Reversed ^= true;
    }
    return false;
}

bool in_hash() {
    return HashSearch(HTable, Hash, CurrentCost);
}

void add_to_hash() {
    HashInsert(HTable, Hash, CurrentCost);
}

void lkh_end() {
    PredSucCostAvailable = 0;
    NormalizeNodeList();
    NormalizeSegmentList();
    Reversed = 0;
}

double current_penalty() {
    return CurrentPenalty;
}

double current_cost() {
    return CurrentCost;
}

void find_tour_begin() {
    for (int64_t i = 1; i < Dimension; ++i) {
        auto &n = NodeSet[i];
        n.OldPred = n.OldSuc = n.NextBestSuc = n.BestSuc = 0;
    }
    BetterCost = INFINITY;
    BetterPenalty = CurrentPenalty = INFINITY;
    HashInitialize(HTable);
}

void find_tour_merge_best() {
    if (FirstNode->BestSuc && !TSPTW_Makespan) {
        /* Merge tour with current best tour */
        auto *t = FirstNode;
        while ((t = t->Next = t->BestSuc) != FirstNode)
            ;
        CurrentCost = MergeWithTour();
    }
}

bool find_tour_record_better() {
    if (std::tie(CurrentPenalty, CurrentCost) < std::tie(BetterPenalty, BetterCost)) {
        BetterCost = CurrentCost;
        BetterPenalty = CurrentPenalty;
        RecordBetterTour();
        AdjustCandidateSet();
        HashInitialize(HTable);
        HashInsert(HTable, Hash, CurrentCost);
        return true;
    }
    return false;
}

void find_tour_end() {
    auto *t = FirstNode;
    if (Norm == 0 || !t->BestSuc) {
        do
            t = t->BestSuc = t->Suc;
        while (t != FirstNode);
    }
    Hash = 0;
    do {
        (t->Suc = t->BestSuc)->Pred = t;
        Hash ^= Rand[t->Id] * Rand[t->Suc->Id];
    } while ((t = t->BestSuc) != FirstNode);
    ResetCandidateSet();
    CurrentPenalty = BetterPenalty;
}

PYBIND11_MODULE(pylkh, m) {
    Err = err_2;
    m.doc() = "LKH3 python binding";
    m.attr("__version__") = VERSION_INFO;
    m.def("init", init_parameters,
          "Initialization.",
          "seed"_a = 43, "trace_level"_a = 1);
    m.def("read_problem", ReadProblem,
          "Read problem definition.",
          "file_path"_a);
    m.def("init_problem", init_problem,
          "Manually init problem.",
          "type"_a, "dimension"_a, "num_vehicle"_a, "capacity"_a, "dist_mat"_a, "time_mat"_a, "demand_mat"_a);
    m.def("get_distance_matrix", get_distance_matrix,
          "Get distance matrix.");
    m.def("get_time_matrix", get_time_matrix,
          "Get time matrix.");
    m.def("preprocess", preprocess,
          "Allocate resources and build candidate set.");
    m.def("set_trial", set_trial,
          "Set the value of Trial, starting from 1. There are special cases for Trial=1, so be careful.",
          "trial"_a);
    m.def("get_trial", get_trial);
    m.def("generate_solution", generate_solution,
          "Generate an initial solution.");
    m.def("get_solution", get_solution,
          "Get current solution sequence.");
    m.def("set_rl_solution", set_rl_solution,
          "Set solution in RL format.",
          "solution"_a);
    m.def("set_dummy_edge", set_dummy_edge,
          "Set edge from node to dummy node.");
    m.def("get_rl_solution", get_rl_solution,
          "Get solution in RL format.");
    m.def("lkh_begin", lkh_begin);
    m.def("lkh_active_positions", lkh_active_positions);
    m.def("lkh_pop_active_pos", lkh_pop_active_pos);
    m.def("lkh_activate", lkh_activate);
    m.def("lkh_try", lkh_try);
    m.def("lkh_apply", lkh_apply);
    m.def("in_hash", in_hash);
    m.def("add_to_hash", add_to_hash);
    m.def("lkh_end", lkh_end);
    m.def("current_penalty", current_penalty);
    m.def("current_cost", current_cost);
    m.def("find_tour_begin", find_tour_begin);
    m.def("find_tour_merge_best", find_tour_merge_best);
    m.def("find_tour_record_better", find_tour_record_better);
    m.def("find_tour_end", find_tour_end);
}
