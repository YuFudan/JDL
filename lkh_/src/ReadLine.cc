#include "LKH.h"

/*
 * The ReadLine function reads the next input line from a file. The function
 * handles the problem that an input line may be terminated by a carriage
 * return, a newline, both, or EOF.
 */

static char *Buffer;
static int64_t MaxBuffer;

static int64_t EndOfLine(FILE *InputFile, int64_t c) {
    int64_t EOL = (c == '\r' || c == '\n');
    if (c == '\r') {
        c = fgetc(InputFile);
        if (c != '\n' && c != EOF)
            ungetc(c, InputFile);
    }
    return EOL;
}

char *ReadLine(FILE *InputFile) {
    int64_t i, c;

    if (Buffer == 0) {
        MaxBuffer = 80;
        Buffer = new char[MaxBuffer]();
    }
    for (i = 0; (c = fgetc(InputFile)) != EOF && !EndOfLine(InputFile, c);
         i++) {
        if (i >= MaxBuffer - 1) {
            MaxBuffer *= 2;
            Buffer = (char *)realloc(Buffer, MaxBuffer);
        }
        Buffer[i] = (char)c;
    }
    Buffer[i] = '\0';
    if (!LastLine || (int64_t)strlen(LastLine) < i) {
        free(LastLine);
        LastLine = new char[i + 1]();
    }
    strcpy(LastLine, Buffer);
    return c == EOF && i == 0 ? 0 : Buffer;
}
