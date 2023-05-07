#include "LKH.h"

/*
 * The WriteCandidates function writes the candidate edges to file
 * CandidateFileName[0].
 *
 * The first line of the file contains the number of nodes.
 *
 * Each of the follwong lines contains a node number, the number of the
 * dad of the node in the minimum spanning tree (0, if the node has no dad),
 * the number of candidate edges emanating from the node, followed by the
 * candidate edges. For each candidate edge its end node number and
 * alpha-value are given.
 *
 * The function is called from the CreateCandidateSet function.
 */

void WriteCandidates() {
    FILE *CandidateFile;
    int64_t i, Count;

    if (CandidateFiles == 0 ||
        !(CandidateFile = fopen(CandidateFileName[0], "w")))
        return;
    if (TraceLevel >= 1)
        printff("Writing CANDIDATE_FILE: \"%s\" ... ",
                CandidateFileName[0]);
    fprintf(CandidateFile, "%ld\n", Dimension);
    for (i = 1; i <= Dimension; i++) {
        auto &N = NodeSet[i];
        fprintf(CandidateFile, "%ld %ld", N.Id, N.Dad ? N.Dad->Id : 0);
        Count = 0;
        for (auto *NN = N.CandidateSet.data(); NN && NN->To; NN++)
            Count++;
        fprintf(CandidateFile, " %ld ", Count);
        for (auto *NN = N.CandidateSet.data(); NN && NN->To; NN++)
            fprintf(CandidateFile, "%ld %ld ", NN->To->Id, NN->Alpha);
        fprintf(CandidateFile, "\n");
    }
    fprintf(CandidateFile, "-1\nEOF\n");
    fclose(CandidateFile);
    if (TraceLevel >= 1)
        printff("done\n");
}
