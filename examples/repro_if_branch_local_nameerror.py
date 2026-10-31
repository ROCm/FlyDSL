#!/usr/bin/env python3

from flydsl.compiler.ast_rewriter import ASTRewriter


def repro_case():
    num_tiles = 3
    if (num_tiles % 2) == 1:
        a0 = 10
    else:
        a0 = 20
    return a0 + 1


if __name__ == "__main__":
    ASTRewriter.transform(repro_case)
    print(repro_case())
