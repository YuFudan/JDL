#include "LKH.h"

#include <exception>

#include "Genetic.h"
#include "Sequence.h"
#include "gpx.h"

// All global variables of the program.

// LKH.h variables
bool AlwaysWriteOutput;
bool WriteSolutionToLog;
int l_opt_limit;
double waiting_penalty;
bool SingleCourier;
bool ReadStdin;
// Number of candidate edges to be associated with each node during the ascent
int64_t AscentCandidates;
// Is the instance asymmetric?
int64_t Asymmetric;
// Number of backbone trials in each run
int64_t BackboneTrials;
// Specifies whether backtracking is used for the first move in a sequence of moves
int64_t Backtracking;
// Cost of the tour in BestTour
GainType BestCost;
// Penalty of the tour in BestTour
GainType BestPenalty;
// Table containing best tour found
int64_t *BestTour;
// Cost of the tour stored in BetterTour
GainType BetterCost;
// Penalty of the tour stored in BetterTour
GainType BetterPenalty;
// Table containing the currently best tour in a run
int64_t *BetterTour;
// Number of black nodes in a BWTSP instance
int64_t BWTSP_B;
// Maximum number of subsequent white nodes in a BWTSP instance
int64_t BWTSP_Q;
// Maximum length of any path between two black nodes in a BTWSP instance
int64_t BWTSP_L;
// Mask for indexing the cache
int64_t CacheMask;
// Table of cached distances
int64_t *CacheVal;
// Table of the signatures of cached distances
int64_t *CacheSig;
// Number of CANDIDATE_FILEs
int64_t CandidateFiles;
// Number of EDGE_FILEs
int64_t EdgeFiles;
// Cost matrix
int64_t *CostMatrix;
// Cost time matrix
double *CostTimeMatrix;
// Cost distance matrix
double *CostDistanceMatrix;
GainType CurrentCost;
GainType CurrentGain;
GainType CurrentPenalty;
GainType TruePenalty;
// Number of commodities in a M-PDTSP instance
int64_t DemandDimension;
Node *Depot;
// 问题定义中的维度，一般是 客户数+1仓库
int64_t Dim_original = -1;
// 按车辆数增加仓库后的维度，等于 Dim_original + salesman - 1
int64_t Dim_salesman = -1;
// Node数目，对于非对称问题等于Dim_salesman*2
int64_t Dimension = -1;
// Maximum route distance for a CVRP instance
double DistanceLimit;
// Maximum alpha-value allowed for any candidate edge is set to Excess times the absolute value of the lower bound of a solution tour
double Excess;
// Number of extra neighbors to be added to the candidate set of each node
int64_t ExtraCandidates;
// First and last node in the list of "active" nodes
Node *FirstActive, *LastActive;
// First constraint in the list of SOP precedence constraints
Constraint *FirstConstraint;
// First node in the list of nodes
Node *FirstNode;
// A pointer to the first segment in the cyclic list of segments
Segment *FirstSegment;
// A pointer to the first super segment in the cyclic list of segments
SSegment *FirstSSegment;
// Specifies whether Gain23 is used
int64_t Gain23Used;
// Specifies whether L&K's gain criterion is used
int64_t GainCriterionUsed;
// Grid size used in toroidal instances
double GridSize;
// Desired initial size of each segment
int64_t GroupSize;
// Desired initial size of each super segment
int64_t SGroupSize;
// Current number of segments
int64_t Groups;
// Current number of super segments
int64_t SGroups;
// Hash value corresponding to the current tour
unsigned Hash;
// Heap used for computing minimum spanning trees
Node **Heap;
// Hash table used for storing tours
HashTable *HTable;
// Length of the first period in the ascent
int64_t InitialPeriod;
// Initial step size used in the ascent
int64_t InitialStepSize;
// Fraction of the initial tour to be constructed by INITIAL_TOUR_FILE edges
double InitialTourFraction;
// Specifies the number of K-swap-kicks
int64_t Kicks;
// Specifies K for a K-swap-kick
int64_t KickType;
// Last input line
char *LastLine;
// Lower bound found by the ascent
double LowerBound;
// The M-value is used when solving an ATSP- instance by transforming it to a STSP-instance
int64_t MaxDistance;
// The maximum number of candidate edges considered at each level of the search for a move
int64_t MaxBreadth;
// Maximum number of candidate edges to be associated with each node
int64_t MaxCandidates;
// Maximum dimension for an explicit cost matrix
int64_t MaxMatrixDimension;
// Maximum number of swaps made during the search for a move
int64_t MaxSwaps;
// Maximum number of trials in each run
int64_t MaxTrials;
// Number of MERGE_TOUR_FILEs
int64_t MergeTourFiles;
// Specifies the sequantial move type to be used in local search. A value K >= 2 signifies that a k-opt moves are tried for k <= K
int64_t MoveType;
// A special (3- or 5-opt) move is used
int64_t MoveTypeSpecial;
// Node列表，[1, Dimension]
std::vector<Node> NodeSet;
// Measure of a 1-tree's discrepancy from a tour
int64_t Norm;
// Specifies the nonsequential move type to be used in local search. A value L >= 4 signifies that nonsequential l-opt moves are tried for l <= L
int64_t NonsequentialMoveType;
// Known optimal tour length. If StopAtOptimum is 1, a run will be terminated as soon as a tour length becomes equal this value
GainType Optimum;
// Specifies the maximum number of alternating cycles to be used for patching disjunct cycles
int64_t PatchingA;
// Specifies the maximum number of disjoint cycles to be patched (by one or more alternating cycles)
int64_t PatchingC;
GainType PenaltyGain;
// PredCost and SucCost are available
int64_t PredSucCostAvailable;
// Specifies whether the first POPMUSIC tour is used as initial tour for LK
int64_t POPMUSIC_InitialTour;
// Maximum number of nearest neighbors used as candidates in iterated 3-opt
int64_t POPMUSIC_MaxNeighbors;
// The sample size
int64_t POPMUSIC_SampleSize;
// Number of solutions to generate
int64_t POPMUSIC_Solutions;
// Maximum trials used for iterated 3-opt
int64_t POPMUSIC_Trials;
// Profits in TRPP
int64_t *Profits = NULL;
// Table of random values
unsigned *Rand;
// IPT or GPX2
int64_t Recombination;
// Specifies whether the choice of the first edge to be broken is restricted
int64_t RestrictedSearch;
// Boolean used to indicate whether a tour has been reversed
bool Reversed;
// Current run number
int64_t Run;
// Total number of runs
int64_t Runs;
// Scale factor for Euclidean and ATT instances
int64_t Scale;
// Service time for a CVRP instance
double ServiceTime;
int64_t Serial;
// Initial seed for random number generation
unsigned Seed;
// Time when execution starts
double StartTime;
// Specifies whether a run will be terminated if the tour length becomes equal to Optimum
int64_t StopAtOptimum;
// Specifies whether the Pi-values should be determined by subgradient optimization
int64_t Subgradient;
// Number of nodes in a subproblem
int64_t SubproblemSize;
// Specifies the move type to be used for all moves following the first move in a sequence of moves. The value K >= 2 signifies that a K-opt move is to be used
int64_t SubsequentMoveType;
// A special (3- or 5-opt) subsequent move is used
int64_t SubsequentMoveTypeSpecial;
// Species whether patching is used for subsequent moves
int64_t SubsequentPatching;
// Stack of SwapRecords
SwapRecord *SwapStack;
int64_t Swaps;
// Saved number of swaps
int64_t OldSwaps;
// The time limit in seconds
double TimeLimit;
// Sum of demands for a CVRP instance
int64_t TotalDemand;
// Specifies the level of detail of the output given during the solution process. The value 0 signifies a minimum amount of output. The higher the value is the more information is given
int64_t TraceLevel;
// Ordinal number of the current trial
int64_t Trial;
GainType TSPTW_CurrentMakespanCost;
int64_t TSPTW_Makespan;

// The following variables are read by the functions ReadParameters and ReadProblem:

char *ProblemFileName, *PiFileName, *TourFileName, *OutputTourFileName, *InputTourFileName, **CandidateFileName, **EdgeFileName, *InitialTourFileName, *SubproblemTourFileName, **MergeTourFileName, *MTSPSolutionFileName, *SINTEFSolutionFileName;
char *Name, *EdgeDataFormat, *NodeCoordType, *DisplayDataType;
int64_t CandidateSetSymmetric, CandidateSetType, CoordType, DelaunayPartitioning, DelaunayPure, ExternalSalesmen, ExtraCandidateSetSymmetric, ExtraCandidateSetType, InitialTourAlgorithm, KarpPartitioning, KCenterPartitioning, KMeansPartitioning, MTSPDepot, MTSPMinSize, MTSPMaxSize, MTSPObjective, MoorePartitioning, PatchingAExtended, PatchingARestricted, PatchingCExtended, PatchingCRestricted, ProblemType, RiskThreshold, RohePartitioning, Salesmen, SierpinskiPartitioning, SubproblemBorders, SubproblemsCompressed, WeightType, WeightFormat;
int64_t Capacity;  // 容量

FILE *ParameterFile, *ProblemFile, *PiFile, *InputTourFile, *InitialTourFile, *SubproblemTourFile, **MergeTourFile;
CostFunction Distance, D, C, c, OldDistance;
MoveFunction BestMove, BacktrackMove, BestSubsequentMove;
PenaltyFunction Penalty;
MergeTourFunction MergeWithTour;  // 回路合并函数，默认是MergeWithTourIPT
ErrorHandler Err;

// Genetic.h variables

// The maximum size of the population
int64_t MaxPopulationSize;
// The current size of the population
int64_t PopulationSize;
CrossoverFunction Crossover;
// Array of individuals (solution tours)
int64_t **Population;
// The fitness (tour penalty) of each individual
GainType *PenaltyFitness;
// The fitness (tour cost) of each individual
GainType *Fitness;

// Sequence.h variables

// The sequence of nodes to be used in a move
Node **t;
// The currently best t's
Node **T;
// For saving t when using the BacktrackKOptMove function
Node **tSaved;
// The permutation corresponding to the sequence in which the t's occur on the tour
int64_t *p;
// The inverse permutation of p
int64_t *q;
// Array: incl[i] == j, if (t[i], t[j]) is an inclusion edge
int64_t *incl;
// Array: cycle[i] is cycle number of t[i]
int64_t *cycle;
// For storing the G-values in the BestKOptMove function
GainType *G;
// The value K for the current K-opt move
int64_t K;

// gpx.h variables
int64_t n_cities, n_cand;
int64_t n_partitions_size2, n_partitions_before_fusion, n_partitions_after_fusion1, n_partitions_after_fusion2, n_partitions_after_fusion3;
int64_t n_partitions_after_fusion4, n_partitions_after_fusion5, n_partitions_after_fusionB;
Node **Map2Node;

bool Fixed(Node *a, Node *b) {
    return a->FixedTo1 == b || a->FixedTo2 == b;
}

bool FixedOrCommon(Node *a, Node *b) {
    return Fixed(a, b) || IsCommonEdge(a, b);
}

bool InBestTour(Node *a, Node *b) {
    return a->BestSuc == b || b->BestSuc == a;
}

bool InNextBestTour(Node *a, Node *b) {
    return a->NextBestSuc == b || b->NextBestSuc == a;
}

bool InInputTour(Node *a, Node *b) {
    return a->InputSuc == b || b->InputSuc == a;
}

bool InInitialTour(Node *a, Node *b) {
    return a->InitialSuc == b || b->InitialSuc == a;
}

bool Near(Node *a, Node *b) {
    return a->BestSuc ? InBestTour(a, b) : a->Dad == b || b->Dad == a;
}

void Follow(Node *a, Node *b) {
    Link(a->Pred, a->Suc);
    Link(a, b->Suc);
    Link(b, a);
}

void Precede(Node *a, Node *b) {
    Link(a->Pred, a->Suc);
    Link(b->Pred, a);
    Link(a, b);
}

void err_1(int n) {
    exit(n);
}

class Error : std::exception {
    const char *what() const noexcept override {
        return "Error";
    }
};

void err_2(int) {
    throw(Error());
}