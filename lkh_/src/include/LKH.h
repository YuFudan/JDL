#ifndef _LKH_H
#define _LKH_H

// This header is used by almost all functions of the program.
// It defines macros and specifies data structures and function prototypes.

#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "GainType.h"
#include "Hashing.h"

const int64_t BIG_INT = INT_MAX / 4;

enum Types { TSP,
             ATSP,
             SOP,
             TSPTW,
             HCP,
             CCVRP,
             CVRP,
             ACVRP,
             CVRPTW,
             RCTVRP,
             RCTVRPTW,
             VRPB,
             VRPBTW,
             PDPTW,
             PDTSP,
             PDTSPF,
             PDTSPL,
             VRPSPDTW,
             OVRP,
             ONE_PDTSP,
             MLP,
             M_PDTSP,
             M1_PDTSP,
             TSPDL,
             TSPPD,
             TOUR,
             HPP,
             BWTSP,
             TRP,
             TRPP,
             CTSP,
             STTSP,
             ADCVRP
};
enum CoordTypes { TWOD_COORDS,
                  THREED_COORDS,
                  NO_COORDS };
enum EdgeWeightTypes { EXPLICIT,
                       EUC_2D,
                       EUC_3D,
                       MAX_2D,
                       MAX_3D,
                       MAN_2D,
                       MAN_3D,
                       CEIL_2D,
                       CEIL_3D,
                       FLOOR_2D,
                       FLOOR_3D,
                       GEO,
                       GEOM,
                       GEO_MEEUS,
                       GEOM_MEEUS,
                       ATT,
                       TOR_2D,
                       TOR_3D,
                       XRAY1,
                       XRAY2,
                       SPECIAL
};
enum EdgeWeightFormats { FUNCTION,
                         FULL_MATRIX,
                         UPPER_ROW,
                         LOWER_ROW,
                         UPPER_DIAG_ROW,
                         LOWER_DIAG_ROW,
                         UPPER_COL,
                         LOWER_COL,
                         UPPER_DIAG_COL,
                         LOWER_DIAG_COL
};
enum CandidateSetTypes { ALPHA,
                         DELAUNAY,
                         NN,
                         POPMUSIC,
                         QUADRANT };
enum InitialTourAlgorithms { BORUVKA,
                             CTSP_ALG,
                             CVRP_ALG,
                             GREEDY,
                             MOORE,
                             MTSP_ALG,
                             NEAREST_NEIGHBOR,
                             QUICK_BORUVKA,
                             SIERPINSKI,
                             SOP_ALG,
                             TSPDL_ALG,
                             WALK
};
enum Objectives { MINMAX,
                  MINMAX_SIZE,
                  MINSUM };
enum RecombinationTypes { IPT,
                          GPX2 };

typedef struct Node Node;
typedef struct Candidate Candidate;
typedef struct Segment Segment;
typedef struct SSegment SSegment;
typedef struct SwapRecord SwapRecord;
typedef struct Constraint Constraint;
typedef Node *(*MoveFunction)(Node *t1, Node *t2, GainType *G0, GainType *Gain);
typedef int64_t (*CostFunction)(Node *Na, Node *Nb);
typedef GainType (*PenaltyFunction)(void);
typedef GainType (*MergeTourFunction)(void);
typedef void (*ErrorHandler)(int);
extern MergeTourFunction MergeWithTour;
extern ErrorHandler Err;

// The Node structure is used to represent nodes (cities) of the problem

struct Node {
    int64_t Id;        // Number of the node (1...Dimension)
    int64_t Index;     // Raw index used for distance and time matrix
    int64_t Loc;       // Location of the node in the heap (zero, if the node is not in the heap)
    int64_t Rank;      // During the ascent, the priority of the node. Otherwise, the ordinal number of the node in the tour
    int64_t V;         // During the ascent the degree of the node minus 2. Otherwise, the variable is used to mark nodes
    int64_t LastV;     // Last value of V during the ascent
    int64_t Cost;      // "Best" cost of an edge emanating from the node
    int64_t NextCost;  // During the ascent, the next best cost of an edge emanating from the node
    int64_t PredCost,  // The costs of the neighbor edges on the current tour
        SucCost;
    int64_t SavedCost;
    int64_t Pi;              // Pi-value of the node
    int64_t BestPi;          // Currently best pi-value found during the ascent
    int64_t Beta;            // Beta-value (used for computing alpha-values)
    int64_t Subproblem;      // Number of the subproblem the node is part of
    int64_t Sons;            // Number of sons in the minimum spanning tree
    int64_t *C;              // (不要使用) 整数距离矩阵
    double *CD;              // 距离矩阵
    double *CT;              // 时间矩阵
    int64_t Special;         // Is the node a special node in Jonker and Volgenant's mTSP to TSP transformation?
    int64_t Demand;          // Customer demand in a CVRP or 1-PDTSP instance
    int64_t *M_Demand;       // Table of demands in an M-PDTSP instance
    int64_t Seq;             // Sequence number in the current tour
    int64_t DraftLimit;      // Draft limit in a TSPDL instance
    int64_t Load;            // Accumulated load in the current route
    int64_t OriginalId;      // The original Id in a SDVRP or STTSPinstance
    Node *Pred, *Suc;        // Predecessor and successor node in the two-way list of nodes
    Node *OldPred, *OldSuc;  // Previous values of Pred and Suc
    Node *BestSuc,           // Best and next best successor node in the
        *NextBestSuc;        // currently best tour
    Node *Dad;               // Father of the node in the minimum 1-tree
    Node *Nearest;           // Nearest node (used in the greedy heuristics)
    Node *Next;              // Auxiliary pointer, usually to the next node in a list of nodes (e.g., the list of "active" nodes)
    Node *Prev;              // Auxiliary pointer, usually to the previous node in a list of nodes
    Node *Mark;              // Visited mark
    Node *FixedTo1,          // Pointers to the opposite end nodes of fixed edges.
        *FixedTo2;           // A maximum of two fixed edges can be incident to a node
    Node *FixedTo1Saved,     // Saved values of FixedTo1 and FixedTo2
        *FixedTo2Saved;
    Node *Head;                           // Head of a segment of fixed or common edges
    Node *Tail;                           // Tail of a segment of fixed or common edges
    Node *InputSuc;                       // Successor in the INPUT_TOUR file
    Node *InitialSuc;                     // Successor in the INITIAL_TOUR file
    Node *SubproblemPred;                 // Predecessor in the SUBPROBLEM_TOUR file
    Node *SubproblemSuc;                  // Successor in the SUBPROBLEM_TOUR file
    Node *SubBestPred;                    // The best predecessor node in a subproblem
    Node *SubBestSuc;                     // The best successor node in a subproblem
    Node *MergePred;                      // Predecessor in the first MERGE_TOUR file
    Node **MergeSuc;                      // Successors in the MERGE_TOUR files
    Node *Added1, *Added2;                // Pointers to the opposite end nodes of added edges in a submove
    Node *Deleted1, *Deleted2;            // Pointers to the opposite end nodes of deleted edges in a submove
    Node *SucSaved;                       // Saved pointer to successor node
    std::vector<Candidate> CandidateSet;  // Candidate array
    Candidate *BackboneCandidateSet;      // Backbone candidate array
    Segment *Parent;                      // Parent segment of a node when the two-level tree representation is used
    Constraint *FirstConstraint;
    int64_t *PathLength;
    int64_t **Path;
    double ServiceTime;
    int64_t Pickup, Delivery;
    int64_t DepotId;  // Equal to Id if the node is a depot; otherwize 0
    double Earliest, Latest;
    int64_t Backhaul;
    int64_t Serial;
    int64_t Color;
    double X, Y, Z;                        // Coordinates of the node
    double Xc, Yc, Zc;                     // Converted coordinates
    char Axis;                             // The axis partitioned when the node is part of a KDTree
    char OldPredExcluded, OldSucExcluded;  // Booleans used for indicating whether one (or both) of the adjoining nodes on the old tour has been excluded
    char Required;                         // Is the node required in a STTSP?
};

// The Candidate structure is used to represent candidate edges

struct Candidate {
    Node *To;       // The end node of the edge
    int64_t Cost;   // Cost (distance) of the edge
    int64_t Alpha;  // Its alpha-value
};

// The Segment structure is used to represent the segments in the two-level representation of tours

struct Segment {
    bool Reversed;        // Reversal bit
    Node *First, *Last;   // First and last node in the segment
    Segment *Pred, *Suc;  // Predecessor and successor in the two-way list of segments
    int64_t Rank;         // Ordinal number of the segment in the list
    int64_t Size;         // Number of nodes in the segment
    SSegment *Parent;     // The parent super segment
};

struct SSegment {
    bool Reversed;          // Reversal bit
    Segment *First, *Last;  // The first and last node in the segment
    SSegment *Pred, *Suc;   // The predecessor and successor in the two-way list of super segments
    int64_t Rank;           // The ordinal number of the segment in the list
    int64_t Size;           // The number of nodes in the segment
};

// The SwapRecord structure is used to record 2-opt moves (swaps)

struct SwapRecord {
    Node *t1, *t2, *t3, *t4;  // The 4 nodes involved in a 2-opt move
};

// The Constraint structure ts used to represent a precedence constraint for a SOP instance

struct Constraint {
    Node *t1, *t2;     // Node t1 has to precede node t2
    Constraint *Suc;   // The next constraint in the list of all constraints
    Constraint *Next;  // The next constraint in the list of constraints for a node
};

extern bool AlwaysWriteOutput;      // Write solution even if penalty!=0
extern bool WriteSolutionToLog;     // Write solution to stdout
extern int l_opt_limit;             // limit of λ-opt
extern double waiting_penalty;      // Custom waiting time penalty for VRPSPDTW
extern bool SingleCourier;          // If single courier, then do not reset clock after each route for VRPSPDTW
extern bool ReadStdin;              // Read from stdin
extern int64_t AscentCandidates;    // Number of candidate edges to be associated with each node during the ascent
extern int64_t Asymmetric;          // Is the instance asymmetric?
extern int64_t BackboneTrials;      // Number of backbone trials in each run
extern int64_t Backtracking;        // Specifies whether backtracking is used for the first move in a sequence of moves
extern GainType BestCost;           // Cost of the tour in BestTour
extern GainType BestPenalty;        // Penalty of the tour in BestTour
extern int64_t *BestTour;           // Table containing best tour found
extern GainType BetterCost;         // Cost of the tour stored in BetterTour
extern GainType BetterPenalty;      // Penalty of the tour stored in BetterTour
extern int64_t *BetterTour;         // Table containing the currently best tour in a run
extern int64_t BWTSP_B;             // Number of black nodes in a BWTSP instance
extern int64_t BWTSP_Q;             // Maximum number of subsequent white nodes in a BWTSP instance
extern int64_t BWTSP_L;             // Maximum length of any path between two black nodes in a BTWSP instance
extern int64_t CacheMask;           // Mask for indexing the cache
extern int64_t *CacheVal;           // Table of cached distances
extern int64_t *CacheSig;           // Table of the signatures of cached distances
extern int64_t CandidateFiles;      // Number of CANDIDATE_FILEs
extern int64_t EdgeFiles;           // Number of EDGE_FILEs
extern int64_t *CostMatrix;         // Cost matrix
extern double *CostTimeMatrix;      // Cost time matrix
extern double *CostDistanceMatrix;  // Cost distance matrix
extern GainType CurrentCost;
extern GainType CurrentGain;
extern GainType CurrentPenalty;
extern GainType TruePenalty;
extern int64_t DemandDimension;  // Number of commodities in a M-PDTSP instance
extern Node *Depot;
extern int64_t Dimension;
extern int64_t Dim_salesman;
extern int64_t Dim_original;
extern double DistanceLimit;
extern double Excess;                   // Maximum alpha-value allowed for any candidate edge is set to Excess times the absolute value of the lower bound of a solution tour
extern int64_t ExtraCandidates;         // Number of extra neighbors to be added to the candidate set of each node
extern Node *FirstActive, *LastActive;  // First and last node in the list of "active" nodes
extern Constraint *FirstConstraint;     // First constraint in the list of SOP precedence constraints
extern Node *FirstNode;                 // First node in the list of nodes
extern Segment *FirstSegment;           // A pointer to the first segment in the cyclic list of segments
extern SSegment *FirstSSegment;         // A pointer to the first super segment in the cyclic list of segments
extern int64_t Gain23Used;              // Specifies whether Gain23 is used
extern int64_t GainCriterionUsed;       // Specifies whether L&K's gain criterion is used
extern double GridSize;                 // Grid size used in toroidal instances
extern int64_t GroupSize;               // Desired initial size of each segment
extern int64_t SGroupSize;              // Desired initial size of each super segment
extern int64_t Groups;                  // Current number of segments
extern int64_t SGroups;                 // Current number of super segments
extern unsigned Hash;                   // Hash value corresponding to the current tour
extern Node **Heap;                     // Heap used for computing minimum spanning trees
extern HashTable *HTable;               // Hash table used for storing tours
extern int64_t InitialPeriod;           // Length of the first period in the ascent
extern int64_t InitialStepSize;         // Initial step size used in the ascent
extern double InitialTourFraction;      // Fraction of the initial tour to be constructed by INITIAL_TOUR_FILE edges
extern int64_t Kicks;                   // Specifies the number of K-swap-kicks
extern int64_t KickType;                // Specifies K for a K-swap-kick
extern char *LastLine;                  // Last input line
extern double LowerBound;               // Lower bound found by the ascent
extern int64_t MaxDistance;             // The M-value is used when solving an ATSP- instance by transforming it to a STSP-instance
extern int64_t MaxBreadth;              // The maximum number of candidate edges considered at each level of the search for a move
extern int64_t MaxCandidates;           // Maximum number of candidate edges to be associated with each node
extern double DistanceLimit;            // Maxixim route distance for a CVRP instance
extern int64_t MaxMatrixDimension;      // Maximum dimension for an explicit cost matrix
extern int64_t MaxSwaps;                // Maximum number of swaps made during the search for a move
extern int64_t MaxTrials;               // Maximum number of trials in each run
extern int64_t MergeTourFiles;          // Number of MERGE_TOUR_FILEs
extern int64_t MoveType;                // Specifies the sequential move type to be used in local search. A value K >= 2 signifies that a k-opt moves are tried for k <= K
extern int64_t MoveTypeSpecial;         // A special (3- or 5-opt) move is used
extern std::vector<Node> NodeSet;
extern int64_t Norm;                   // Measure of a 1-tree's discrepancy from a tour
extern int64_t NonsequentialMoveType;  // Specifies the nonsequential move type to be used in local search. A value L >= 4 signifies that nonsequential l-opt moves are tried for l <= L
extern GainType Optimum;               // Known optimal tour length. If StopAtOptimum is 1, a run will be terminated as soon as a tour length becomes equal this value
extern int64_t PatchingA;              // Specifies the maximum number of alternating cycles to be used for patching disjunct cycles
extern int64_t PatchingC;              // Specifies the maximum number of disjoint cycles to be patched (by one or more alternating cycles)
extern GainType PenaltyGain;
extern int64_t PredSucCostAvailable;   // PredCost and SucCost are available
extern int64_t POPMUSIC_InitialTour;   // Specifies whether the first POPMUSIC tour is used as initial tour for LK
extern int64_t POPMUSIC_MaxNeighbors;  // Maximum number of nearest neighbors used as candidates in iterated 3-opt
extern int64_t POPMUSIC_SampleSize;    // The sample size
extern int64_t POPMUSIC_Solutions;     // Number of solutions to generate
extern int64_t POPMUSIC_Trials;        // Maximum trials used for iterated 3-opt
extern int64_t *Profits;               // Profits in TRPP
extern unsigned *Rand;                 // Table of random values
extern int64_t Recombination;          // IPT or GPX2
extern int64_t RestrictedSearch;       // Specifies whether the choice of the first edge to be broken is restricted
extern bool Reversed;                  // Boolean used to indicate whether a tour has been reversed
extern int64_t Run;                    // Current run number
extern int64_t Runs;                   // Total number of runs
extern int64_t Scale;                  // Scale factor for Euclidean and ATT instances
extern double ServiceTime;             // Service time for a CVRP instance
extern int64_t Serial;
extern unsigned Seed;                      // Initial seed for random number generation
extern double StartTime;                   // Time when execution starts
extern int64_t StopAtOptimum;              // Specifies whether a run will be terminated if the tour length becomes equal to Optimum
extern int64_t Subgradient;                // Specifies whether the Pi-values should be determined by subgradient optimization
extern int64_t SubproblemSize;             // Number of nodes in a subproblem
extern int64_t SubsequentMoveType;         // Specifies the move type to be used for all moves following the first move in a sequence of moves. The value K >= 2 signifies that a K-opt move is to be used
extern int64_t SubsequentMoveTypeSpecial;  // A special (3- or 5-opt) subsequent move is used
extern int64_t SubsequentPatching;         // Species whether patching is used for subsequent moves
extern SwapRecord *SwapStack;              // Stack of SwapRecords
extern int64_t Swaps;                      // Flip操作的次数
extern int64_t OldSwaps;                   // Saved number of swaps
extern double TimeLimit;                   // The time limit in seconds
extern int64_t TotalDemand;                // Sum of demands for a CVRP instance
extern int64_t TraceLevel;                 // Specifies the level of detail of the output given during the solution process. The value 0 signifies a minimum amount of output. The higher the value is the more information is given
extern int64_t Trial;                      // Ordinal number of the current trial
extern GainType TSPTW_CurrentMakespanCost;
extern int64_t TSPTW_Makespan;

// The following variables are read by the functions ReadParameters and ReadProblem :

extern char *ProblemFileName,
    *PiFileName,
    *TourFileName,
    *OutputTourFileName,
    *InputTourFileName,
    **CandidateFileName,
    **EdgeFileName,
    *InitialTourFileName,
    *SubproblemTourFileName,
    **MergeTourFileName,
    *MTSPSolutionFileName,
    *SINTEFSolutionFileName;
extern char *Name, *EdgeWeightType, *EdgeWeightFormat,
    *EdgeDataFormat, *NodeCoordType, *DisplayDataType;
extern int64_t CandidateSetSymmetric, CandidateSetType, Capacity,
    CoordType, DelaunayPartitioning, DelaunayPure,
    ExternalSalesmen,
    ExtraCandidateSetSymmetric, ExtraCandidateSetType,
    InitialTourAlgorithm,
    KarpPartitioning, KCenterPartitioning, KMeansPartitioning,
    MTSPDepot, MTSPMinSize, MTSPMaxSize, MTSPObjective,
    MoorePartitioning,
    PatchingAExtended, PatchingARestricted,
    PatchingCExtended, PatchingCRestricted,
    ProblemType, RiskThreshold,
    RohePartitioning, Salesmen, SierpinskiPartitioning,
    SubproblemBorders, SubproblemsCompressed, WeightType, WeightFormat;

extern FILE *ParameterFile, *ProblemFile, *PiFile, *InputTourFile,
    *InitialTourFile, *SubproblemTourFile, **MergeTourFile;
extern CostFunction Distance, D, C, c, OldDistance;
extern MoveFunction BestMove, BacktrackMove, BestSubsequentMove;
extern PenaltyFunction Penalty;

// Function prototypes:

int64_t Distance_1(Node *Na, Node *Nb);
int64_t Distance_LARGE(Node *Na, Node *Nb);
int64_t Distance_Asymmetric(Node *Na, Node *Nb);
int64_t Distance_ATSP(Node *Na, Node *Nb);
int64_t Distance_ATT(Node *Na, Node *Nb);
int64_t Distance_CEIL_2D(Node *Na, Node *Nb);
int64_t Distance_CEIL_3D(Node *Na, Node *Nb);
int64_t Distance_EXPLICIT(Node *Na, Node *Nb);
int64_t Distance_EUC_2D(Node *Na, Node *Nb);
int64_t Distance_EUC_3D(Node *Na, Node *Nb);
int64_t Distance_FLOOR_2D(Node *Na, Node *Nb);
int64_t Distance_FLOOR_3D(Node *Na, Node *Nb);
int64_t Distance_GEO(Node *Na, Node *Nb);
int64_t Distance_GEOM(Node *Na, Node *Nb);
int64_t Distance_GEO_MEEUS(Node *Na, Node *Nb);
int64_t Distance_GEOM_MEEUS(Node *Na, Node *Nb);
int64_t Distance_MAN_2D(Node *Na, Node *Nb);
int64_t Distance_MAN_3D(Node *Na, Node *Nb);
int64_t Distance_MAX_2D(Node *Na, Node *Nb);
int64_t Distance_MAX_3D(Node *Na, Node *Nb);
int64_t Distance_MTSP(Node *Na, Node *Nb);
int64_t Distance_SOP(Node *Na, Node *Nb);
int64_t Distance_SPECIAL(Node *Na, Node *Nb);
int64_t Distance_TOR_2D(Node *Na, Node *Nb);
int64_t Distance_TOR_3D(Node *Na, Node *Nb);
int64_t Distance_XRAY1(Node *Na, Node *Nb);
int64_t Distance_XRAY2(Node *Na, Node *Nb);

int64_t D_EXPLICIT(Node *Na, Node *Nb);
int64_t D_FUNCTION(Node *Na, Node *Nb);

int64_t C_EXPLICIT(Node *Na, Node *Nb);
int64_t C_FUNCTION(Node *Na, Node *Nb);

int64_t c_ATT(Node *Na, Node *Nb);
int64_t c_CEIL_2D(Node *Na, Node *Nb);
int64_t c_CEIL_3D(Node *Na, Node *Nb);
int64_t c_EUC_2D(Node *Na, Node *Nb);
int64_t c_EUC_3D(Node *Na, Node *Nb);
int64_t c_FLOOR_2D(Node *Na, Node *Nb);
int64_t c_FLOOR_3D(Node *Na, Node *Nb);
int64_t c_GEO(Node *Na, Node *Nb);
int64_t c_GEOM(Node *Na, Node *Nb);
int64_t c_GEO_MEEUS(Node *Na, Node *Nb);
int64_t c_GEOM_MEEUS(Node *Na, Node *Nb);

void Activate(Node *t);
bool AddCandidate(Node *From, Node *To, int64_t Cost, int64_t Alpha);
void AddExtraCandidates(int64_t K, int64_t CandidateSetType, int64_t Symmetric);
void AddExtraDepotCandidates(void);
void AddTourCandidates(void);
void AdjustCandidateSet(void);
void AdjustClusters(int64_t K, Node **Center);
void AllocateSegments(void);
void AllocateStructures(void);
GainType Ascent(void);
Node *Best2OptMove(Node *t1, Node *t2, GainType *G0, GainType *Gain);
Node *Best3OptMove(Node *t1, Node *t2, GainType *G0, GainType *Gain);
Node *Best4OptMove(Node *t1, Node *t2, GainType *G0, GainType *Gain);
Node *Best5OptMove(Node *t1, Node *t2, GainType *G0, GainType *Gain);
Node *BestKOptMove(Node *t1, Node *t2, GainType *G0, GainType *Gain);
Node *BestSpecialOptMove(Node *t1, Node *t2, GainType *G0, GainType *Gain);
int64_t Between(const Node *ta, const Node *tb, const Node *tc);
int64_t Between_SL(const Node *ta, const Node *tb, const Node *tc);
int64_t Between_SSL(const Node *ta, const Node *tb, const Node *tc);
GainType BridgeGain(Node *s1, Node *s2, Node *s3, Node *s4, Node *s5, Node *s6, Node *s7, Node *s8, int64_t Case6, GainType G);
Node **BuildKDTree(int64_t Cutoff);
void ChooseInitialTour(void);
void Connect(Node *N1, int64_t Max, int64_t Sparse);
void CandidateReport(void);
void CreateCandidateSet(void);
void CreateDelaunayCandidateSet(void);
void CreateNearestNeighborCandidateSet(int64_t K);
void CreateNNCandidateSet(int64_t K);
void Create_POPMUSIC_CandidateSet(int64_t K);
void CreateQuadrantCandidateSet(int64_t K);
GainType CTSP_InitialTour(void);
GainType CVRP_InitialTour(void);
void eprintf(const char *fmt, ...);
int64_t Excludable(Node *ta, Node *tb);
void Exclude(Node *ta, Node *tb);
int64_t FixedOrCommonCandidates(Node *N);
GainType FindTour(void);
void Flip(Node *t1, Node *t2, Node *t3);
void Flip_SL(Node *t1, Node *t2, Node *t3);
void Flip_SSL(Node *t1, Node *t2, Node *t3);
int64_t Forbidden(Node *Na, Node *Nb);
void FreeCandidateSets(void);
void FreeSegments(void);
void FreeStructures(void);
char *FullName(char *Name, GainType Cost);
int64_t fscanint(FILE *f, int64_t *v);
GainType Gain23(void);
void GenerateCandidates(int64_t MaxCandidates, GainType MaxAlpha, int64_t Symmetric);
double GetTime(void);
GainType GreedyTour(void);
int64_t Improvement(GainType *Gain, Node *t1, Node *SUCt1);
void InitializeStatistics(void);

bool IsBackboneCandidate(const Node *ta, const Node *tb);
bool IsCandidate(const Node *ta, const Node *tb);
bool IsCommonEdge(const Node *ta, const Node *tb);
bool IsPossibleCandidate(Node *From, Node *To);
void KSwapKick(int64_t K);
GainType LinKernighan(void);
void Make2OptMove(Node *t1, Node *t2, Node *t3, Node *t4);
void Make3OptMove(Node *t1, Node *t2, Node *t3, Node *t4, Node *t5, Node *t6, int64_t Case);
void Make4OptMove(Node *t1, Node *t2, Node *t3, Node *t4, Node *t5, Node *t6, Node *t7, Node *t8, int64_t Case);
void Make5OptMove(Node *t1, Node *t2, Node *t3, Node *t4, Node *t5, Node *t6, Node *t7, Node *t8, Node *t9, Node *t10, int64_t Case);
void MakeKOptMove(int64_t K);
GainType MergeTourWithBestTour(void);
GainType MergeWithTourIPT(void);
GainType MergeWithTourGPX2(void);
GainType Minimum1TreeCost(int64_t Sparse);
void MinimumSpanningTree(int64_t Sparse);
void MTSP2TSP(void);
GainType MTSP_InitialTour(void);
int64_t MTSP_DiffSize(void);
void MTSP_WriteSolution(char *FileName, GainType Penalty, GainType Cost);
void MTSP_Report(GainType Penalty, GainType Cost);
void NormalizeNodeList(void);
void NormalizeSegmentList(void);
void OrderCandidateSet(int64_t MaxCandidates, GainType MaxAlpha, int64_t Symmetric);
GainType PatchCycles(int64_t k, GainType Gain);
GainType Penalty_ACVRP(void);
GainType Penalty_BWTSP(void);
GainType Penalty_CCVRP(void);
GainType Penalty_CTSP(void);
GainType Penalty_CVRP(void);
GainType Penalty_CVRPTW(void);
GainType Penalty_MTSP_MINMAX(void);
GainType Penalty_MTSP_MINMAX_SIZE(void);
GainType Penalty_MTSP_MINSUM(void);
GainType Penalty_VRPSPDTW(void);
GainType Penalty_1_PDTSP(void);
GainType Penalty_MLP(void);
GainType Penalty_M_PDTSP(void);
GainType Penalty_M1_PDTSP(void);
GainType Penalty_OVRP(void);
GainType Penalty_PDPTW(void);
GainType Penalty_PDTSP(void);
GainType Penalty_PDTSPF(void);
GainType Penalty_PDTSPL(void);
GainType Penalty_SOP(void);
GainType Penalty_RCTVRP(void);
GainType Penalty_TRP(void);
GainType Penalty_TRPP(void);
GainType Penalty_TSPDL(void);
GainType Penalty_TSPPD(void);
GainType Penalty_TSPTW(void);
GainType Penalty_VRPB(void);
GainType Penalty_VRPBTW(void);
GainType Penalty_VRPSPDTW(void);
void PDPTW_Reduce(void);
void printff(const char *fmt, ...);
void PrintParameters(void);
void PrintStatistics(void);
unsigned Random(void);
int64_t ReadCandidates(int64_t MaxCandidates);
int64_t ReadCandidates(int64_t MaxCandidates);
int64_t ReadEdges(int64_t MaxCandidates);
char *ReadLine(FILE *InputFile);
void ReadParameters(const char *ParameterFileName);
int64_t ReadPenalties(void);
void ReadProblem(const char *problem_file_name);
void ReadTour(char *FileName, FILE **File);
void RecordBestTour(void);
void RecordBetterTour(void);
Node *RemoveFirstActive(void);
void ResetCandidateSet(void);
void RestoreTour(void);
int64_t SegmentSize(Node *ta, Node *tb);
void SINTEF_WriteSolution(char *FileName, GainType Cost);
GainType SFCTour(int64_t CurveType);
void SolveCompressedSubproblem(int64_t CurrentSubproblem, int64_t Subproblems, GainType *GlobalBestCost);
void SolveDelaunaySubproblems(void);
void SolveKarpSubproblems(void);
void SolveKCenterSubproblems(void);
void SolveKMeansSubproblems(void);
void SolveRoheSubproblems(void);
void SolveSFCSubproblems(void);
int64_t SolveSubproblem(int64_t CurrentSubproblem, int64_t Subproblems, GainType *GlobalBestCost);
void SolveSubproblemBorderProblems(int64_t Subproblems, GainType *GlobalCost);
void SolveTourSegmentSubproblems(void);
GainType SOP_InitialTour(void);
GainType SOP_RepairTour(void);
void STTSP2TSP(void);
void SOP_Report(GainType Cost);
int64_t SpecialMove(Node *t1, Node *t2, GainType *G0, GainType *Gain);
void StatusReport(double EntryTime, const char *Suffix);
void StoreTour(void);
void SRandom(unsigned seed);
void SymmetrizeCandidateSet(void);
GainType TSPDL_InitialTour(void);
GainType TSPTW_MakespanCost(void);
void TSPTW_Reduce(void);
void TrimCandidateSet(int64_t MaxCandidates);
void UpdateStatistics(GainType Cost, double Time);
void VRPB_Reduce(void);
void WriteCandidates(void);
void WritePenalties(void);
void WriteTour(char *FileName, int64_t *Tour, GainType Cost);

// 将ab的Suc和Pred连接
template <class T>
void Link(T *a, T *b) {
    a->Suc = b;
    b->Pred = a;
}
// 是FixedTo1/2
bool Fixed(Node *a, Node *b);
bool FixedOrCommon(Node *a, Node *b);
// 是BestSuc
bool InBestTour(Node *a, Node *b);
// 是NextBestSuc
bool InNextBestTour(Node *a, Node *b);
// 是InputSuc
bool InInputTour(Node *a, Node *b);
// 是InitialSuc
bool InInitialTour(Node *a, Node *b);
// 是BestSuc或Dad
bool Near(Node *a, Node *b);
// 把a取出，插入b后方
void Follow(Node *a, Node *b);
// 把a取出，插入b前方
void Precede(Node *a, Node *b);

void err_1(int);
void err_2(int);

void CreateNodes(void);
bool FixEdge(Node *Na, Node *Nb);

#define NOT_IMP \
    { eprintf("Not implemented."); }

#ifdef NDEBUG
#define TO_STR_(x) #x
#define TO_STR(x) TO_STR_(x)
#define THROW(x)                               \
    puts(__FILE__ ":" TO_STR(__LINE__) " " x); \
    Err(-1)
#define ASSERT(x)         \
    if (!(x)) {           \
        THROW(TO_STR(x)); \
    }
#else
#define ASSERT assert
#endif

class Reader {
   private:
    std::vector<char> buffer;
    std::string line;

   public:
    Reader() = default;
    char *read_stdin() {
        std::getline(std::cin, line);
        if (line == "$$$") {
            return nullptr;
        }
        auto n = line.size();
        buffer.resize(n + 2);
        std::copy(line.c_str(), line.c_str() + n, &buffer.front());
        buffer.at(n) = '\n';
        buffer.at(n + 1) = 0;
        return buffer.data();
    }
};

#endif
