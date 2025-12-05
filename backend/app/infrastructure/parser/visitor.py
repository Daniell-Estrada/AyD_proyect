"""
Abstract Syntax Tree (AST) Visitor Pattern Implementation
Provides interface for traversing and processing AST nodes.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.domain.models.ast import (
    AddEdgeFunction,
    AddNodeFunction,
    ArrayAccess,
    ArraySlice,
    ArrayTarget,
    ArrayVarDecl,
    Assignment,
    BinOp,
    Bool,
    CallMethod,
    CallStmt,
    CeilFunction,
    ClassDef,
    Comment,
    ConcatFunction,
    FieldAccess,
    FieldTarget,
    FloorFunction,
    ForLoop,
    ForEachLoop,
    FuncCallExpr,
    GraphOperation,
    GraphTraversal,
    GraphVarDecl,
    IfElse,
    LengthFunction,
    NeighborsFunction,
    NewGraph,
    NewObject,
    Null,
    Number,
    ObjectVarDecl,
    Parameter,
    PrintStmt,
    Program,
    RepeatUntil,
    ReturnStmt,
    ShortCircuitBinOp,
    String,
    StrlenFunction,
    SubroutineDef,
    SubstringFunction,
    UnOp,
    Var,
    VarDecl,
    VarTarget,
    WhileLoop,
)


class ASTVisitor(ABC):
    """Visitor interface for traversing AST nodes"""

    @abstractmethod
    def visit_program(self, node: Program) -> Any:
        pass

    @abstractmethod
    def visit_comment(self, node: Comment) -> Any:
        pass

    @abstractmethod
    def visit_var_decl(self, node: VarDecl) -> Any:
        pass

    @abstractmethod
    def visit_assignment(self, node: Assignment) -> Any:
        pass

    @abstractmethod
    def visit_var_target(self, node: VarTarget) -> Any:
        pass

    @abstractmethod
    def visit_array_target(self, node: ArrayTarget) -> Any:
        pass

    @abstractmethod
    def visit_field_target(self, node: FieldTarget) -> Any:
        pass

    @abstractmethod
    def visit_for_loop(self, node: ForLoop) -> Any:
        pass

    @abstractmethod
    def visit_for_each_loop(self, node: ForEachLoop) -> Any:
        pass

    @abstractmethod
    def visit_while_loop(self, node: WhileLoop) -> Any:
        pass

    @abstractmethod
    def visit_repeat_until(self, node: RepeatUntil) -> Any:
        pass

    @abstractmethod
    def visit_if_else(self, node: IfElse) -> Any:
        pass

    @abstractmethod
    def visit_call_stmt(self, node: CallStmt) -> Any:
        pass

    @abstractmethod
    def visit_call_method(self, node: CallMethod) -> Any:
        pass

    @abstractmethod
    def visit_return_stmt(self, node: ReturnStmt) -> Any:
        pass

    @abstractmethod
    def visit_print_stmt(self, node: PrintStmt) -> Any:
        pass

    @abstractmethod
    def visit_array_access(self, node: ArrayAccess) -> Any:
        pass

    @abstractmethod
    def visit_array_slice(self, node: ArraySlice) -> Any:
        pass

    @abstractmethod
    def visit_field_access(self, node: FieldAccess) -> Any:
        pass

    @abstractmethod
    def visit_func_call_expr(self, node: FuncCallExpr) -> Any:
        pass

    @abstractmethod
    def visit_bin_op(self, node: BinOp) -> Any:
        pass

    @abstractmethod
    def visit_short_circuit_bin_op(self, node: ShortCircuitBinOp) -> Any:
        pass

    @abstractmethod
    def visit_un_op(self, node: UnOp) -> Any:
        pass

    @abstractmethod
    def visit_number(self, node: Number) -> Any:
        pass

    @abstractmethod
    def visit_string(self, node: String) -> Any:
        pass

    @abstractmethod
    def visit_var(self, node: Var) -> Any:
        pass

    @abstractmethod
    def visit_bool(self, node: Bool) -> Any:
        pass

    @abstractmethod
    def visit_null(self, node: Null) -> Any:
        pass

    @abstractmethod
    def visit_class_def(self, node: ClassDef) -> Any:
        pass

    @abstractmethod
    def visit_new_object(self, node: NewObject) -> Any:
        pass

    @abstractmethod
    def visit_subroutine_def(self, node: SubroutineDef) -> Any:
        pass

    @abstractmethod
    def visit_parameter(self, node: Parameter) -> Any:
        pass

    @abstractmethod
    def visit_length_function(self, node: LengthFunction) -> Any:
        pass

    @abstractmethod
    def visit_ceil_function(self, node: CeilFunction) -> Any:
        pass

    @abstractmethod
    def visit_floor_function(self, node: FloorFunction) -> Any:
        pass

    @abstractmethod
    def visit_strlen_function(self, node: StrlenFunction) -> Any:
        pass

    @abstractmethod
    def visit_concat_function(self, node: ConcatFunction) -> Any:
        pass

    @abstractmethod
    def visit_substring_function(self, node: SubstringFunction) -> Any:
        pass

    @abstractmethod
    def visit_new_graph(self, node: NewGraph) -> Any:
        pass

    @abstractmethod
    def visit_add_node_function(self, node: AddNodeFunction) -> Any:
        pass

    @abstractmethod
    def visit_add_edge_function(self, node: AddEdgeFunction) -> Any:
        pass

    @abstractmethod
    def visit_neighbors_function(self, node: NeighborsFunction) -> Any:
        pass

    @abstractmethod
    def visit_array_var_decl(self, node: ArrayVarDecl) -> Any:
        pass

    @abstractmethod
    def visit_object_var_decl(self, node: ObjectVarDecl) -> Any:
        pass

    @abstractmethod
    def visit_graph_var_decl(self, node: GraphVarDecl) -> Any:
        pass

    @abstractmethod
    def visit_graph_operation(self, node: GraphOperation) -> Any:
        pass

    @abstractmethod
    def visit_graph_traversal(self, node: GraphTraversal) -> Any:
        pass


class DefaultASTVisitor(ASTVisitor):
    """Default visitor that simply traverses the AST without doing anything"""

    def visit_program(self, node: Program) -> Any:
        for stmt in node.statements:
            stmt.accept(self)

    def visit_comment(self, node: Comment) -> Any:
        pass

    def visit_var_decl(self, node: VarDecl) -> Any:
        pass

    def visit_assignment(self, node: Assignment) -> Any:
        node.target.accept(self)
        node.value.accept(self)

    def visit_var_target(self, node: VarTarget) -> Any:
        pass

    def visit_array_target(self, node: ArrayTarget) -> Any:
        if isinstance(node.index, list):
            for idx in node.index:
                if hasattr(idx, "accept"):
                    idx.accept(self)
        elif hasattr(node.index, "accept"):
            node.index.accept(self)

    def visit_field_target(self, node: FieldTarget) -> Any:
        pass

    def visit_for_loop(self, node: ForLoop) -> Any:
        node.start.accept(self)
        node.end.accept(self)
        for stmt in node.body:
            stmt.accept(self)

    def visit_for_each_loop(self, node: ForEachLoop) -> Any:
        if hasattr(node.collection, "accept"):
            node.collection.accept(self)
        for stmt in node.body:
            stmt.accept(self)

    def visit_while_loop(self, node: WhileLoop) -> Any:
        node.cond.accept(self)
        for stmt in node.body:
            stmt.accept(self)

    def visit_repeat_until(self, node: RepeatUntil) -> Any:
        for stmt in node.body:
            stmt.accept(self)
        node.cond.accept(self)

    def visit_if_else(self, node: IfElse) -> Any:
        node.cond.accept(self)
        for stmt in node.then_branch:
            stmt.accept(self)
        for stmt in node.else_branch:
            stmt.accept(self)

    def visit_call_stmt(self, node: CallStmt) -> Any:
        for arg in node.args:
            arg.accept(self)

    def visit_call_method(self, node: CallMethod) -> Any:
        node.obj.accept(self)
        for arg in node.args:
            arg.accept(self)

    def visit_return_stmt(self, node: ReturnStmt) -> Any:
        if node.value:
            node.value.accept(self)

    def visit_print_stmt(self, node: PrintStmt) -> Any:
        if node.value and hasattr(node.value, "accept"):
            node.value.accept(self)

    def visit_array_access(self, node: ArrayAccess) -> Any:
        if isinstance(node.index, list):
            for idx in node.index:
                if hasattr(idx, "accept"):
                    idx.accept(self)
        elif hasattr(node.index, "accept"):
            node.index.accept(self)

    def visit_array_slice(self, node: ArraySlice) -> Any:
        if isinstance(node.ranges, list):
            for r in node.ranges:
                if isinstance(r, tuple):
                    s, e = r
                    if s and hasattr(s, "accept"):
                        s.accept(self)
                    if e and hasattr(e, "accept"):
                        e.accept(self)
                else:
                    if hasattr(r, "accept"):
                        r.accept(self)
        else:
            if node.start:
                if hasattr(node.start, "accept"):
                    node.start.accept(self)

            if node.end:
                if hasattr(node.end, "accept"):
                    node.end.accept(self)

    def visit_field_access(self, node: FieldAccess) -> Any:
        pass

    def visit_func_call_expr(self, node: FuncCallExpr) -> Any:
        for arg in node.args:
            arg.accept(self)

    def visit_bin_op(self, node: BinOp) -> Any:
        node.left.accept(self)
        if node.right:
            node.right.accept(self)

    def visit_short_circuit_bin_op(self, node: ShortCircuitBinOp) -> Any:
        node.left.accept(self)
        node.right.accept(self)

    def visit_un_op(self, node: UnOp) -> Any:
        node.value.accept(self)

    def visit_number(self, node: Number) -> Any:
        pass

    def visit_string(self, node: String) -> Any:
        pass

    def visit_var(self, node: Var) -> Any:
        pass

    def visit_bool(self, node: Bool) -> Any:
        pass

    def visit_null(self, node: Null) -> Any:
        pass

    def visit_class_def(self, node: ClassDef) -> Any:
        pass

    def visit_new_object(self, node: NewObject) -> Any:
        pass

    def visit_subroutine_def(self, node: SubroutineDef) -> Any:
        for param in node.parameters:
            param.accept(self)
        for stmt in node.body:
            stmt.accept(self)

    def visit_parameter(self, node: Parameter) -> Any:
        pass

    def visit_length_function(self, node: LengthFunction) -> Any:
        if node.array and hasattr(node.array, "accept"):
            node.array.accept(self)

    def visit_ceil_function(self, node: CeilFunction) -> Any:
        node.expr.accept(self)

    def visit_floor_function(self, node: FloorFunction) -> Any:
        node.expr.accept(self)

    def visit_strlen_function(self, node: StrlenFunction) -> Any:
        if hasattr(node.expr, "accept"):
            node.expr.accept(self)

    def visit_concat_function(self, node: ConcatFunction) -> Any:
        if hasattr(node.left, "accept"):
            node.left.accept(self)
        if hasattr(node.right, "accept"):
            node.right.accept(self)

    def visit_substring_function(self, node: SubstringFunction) -> Any:
        if hasattr(node.string, "accept"):
            node.string.accept(self)
        if hasattr(node.start, "accept"):
            node.start.accept(self)
        if node.length and hasattr(node.length, "accept"):
            node.length.accept(self)

    def visit_new_graph(self, node: NewGraph) -> Any:
        pass

    def visit_graph_operation(self, node: GraphOperation) -> Any:
        for n in getattr(node, "nodes", []):
            if hasattr(n, "accept"):
                n.accept(self)

    def visit_graph_traversal(self, node: GraphTraversal) -> Any:
        if hasattr(node.start_node, "accept"):
            node.start_node.accept(self)
        for op in getattr(node, "operations", []):
            if hasattr(op, "accept"):
                op.accept(self)

    def visit_add_node_function(self, node: AddNodeFunction) -> Any:
        if hasattr(node.node, "accept"):
            node.node.accept(self)

    def visit_add_edge_function(self, node: AddEdgeFunction) -> Any:
        if hasattr(node.from_node, "accept"):
            node.from_node.accept(self)
        if hasattr(node.to_node, "accept"):
            node.to_node.accept(self)

    def visit_neighbors_function(self, node: NeighborsFunction) -> Any:
        if hasattr(node.node, "accept"):
            node.node.accept(self)

    def visit_array_var_decl(self, node: ArrayVarDecl) -> Any:
        for dim in getattr(node, "dimensions", []):
            if hasattr(dim, "accept"):
                dim.accept(self)

    def visit_object_var_decl(self, node: ObjectVarDecl) -> Any:
        pass

    def visit_graph_var_decl(self, node: GraphVarDecl) -> Any:
        pass


class RecursionChecker(DefaultASTVisitor):
    """Visitor to check for recursion in function calls"""

    def __init__(self, target_name: str):
        self.target_name = target_name
        self.found = False

    def visit_func_call_expr(self, node: FuncCallExpr):
        if node.name == self.target_name:
            self.found = True
        super().visit_func_call_expr(node)

    def visit_call_stmt(self, node: CallStmt):
        if node.name == self.target_name:
            self.found = True
        super().visit_call_stmt(node)


class LoopChecker(DefaultASTVisitor):
    """Visitor to check for presence of loops in the AST"""

    def __init__(self):
        self.found = False

    def visit_for_loop(self, node: ForLoop):
        self.found = True
        super().visit_for_loop(node)

    def visit_for_each_loop(self, node: ForEachLoop):
        self.found = True
        super().visit_for_each_loop(node)

    def visit_while_loop(self, node: WhileLoop):
        self.found = True
        super().visit_while_loop(node)

    def visit_repeat_until(self, node: RepeatUntil):
        self.found = True
        super().visit_repeat_until(node)


class WorkAnalyzer(DefaultASTVisitor):
    """Visitor to analyze the maximum nesting level of loops in the AST"""

    def __init__(self):
        self.max_nesting = 0
        self.current_nesting = 0
        self.has_loops = False

    def visit_for_loop(self, node: ForLoop):
        self.has_loops = True
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        super().visit_for_loop(node)
        self.current_nesting -= 1

    def visit_for_each_loop(self, node: ForEachLoop):
        self.has_loops = True
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        super().visit_for_each_loop(node)
        self.current_nesting -= 1

    def visit_while_loop(self, node: WhileLoop):
        self.has_loops = True
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        super().visit_while_loop(node)
        self.current_nesting -= 1
