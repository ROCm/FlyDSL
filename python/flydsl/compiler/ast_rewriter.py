import ast
from typing import List


class ASTRewriter:
    transform_manager: List[ast.NodeTransformer] = []

    @classmethod
    def register(cls, transformer: ast.NodeTransformer):
        cls.transform_manager.append(transformer)

    @classmethod
    def transform(self, func):
        pass
        # for transformer in self.transform_manager:
        #     node = transformer.visit(node)
        # return node
