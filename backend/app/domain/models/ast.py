"""
Abstract Syntax Tree (AST) representation for algorithmic code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class SourcePosition:
    """
    Source code position information for error reporting.
    """

    line: int = 0
    column: int = 0
    end_line: int = 0
    end_column: int = 0
    filename: Optional[str] = None

    def __str__(self) -> str:
        """
        Format position as 'file:line:column' or 'line:column'.
        """
        if self.filename:
            return f"{self.filename}:{self.line}:{self.column}"
        return f"{self.line}:{self.column}"


@dataclass
class ASTMetadata:
    """
    Metadata for AST nodes supporting analysis and optimization.

    Enables attaching analysis hints, complexity annotations, and
    pattern classifications without modifying core AST structure.
    """

    position: Optional[SourcePosition] = None
    complexity_hints: Optional[Dict[str, Any]] = None
    loop_depth: int = 0
    is_recursive: bool = False
    pattern_type: Optional[str] = None


class ASTNode(ABC):
    """
    Base class for all AST nodes implementing Visitor pattern.
    """

    metadata: Optional[ASTMetadata] = None

    def __init__(self):
        self.metadata: Optional[ASTMetadata] = None

    @abstractmethod
    def accept(self, visitor: Any) -> Any:
        """
        Accept visitor for tree traversal (Visitor pattern).
        """

    def set_position(
        self,
        line: int,
        column: int,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
        filename: Optional[str] = None,
    ):
        """
        Set source position information for error reporting.
        """
        if not self.metadata:
            self.metadata = ASTMetadata()

        self.metadata.position = SourcePosition(
            line=line,
            column=column,
            end_line=end_line or line,
            end_column=end_column or column,
            filename=filename,
        )

    def get_position(self) -> Optional[SourcePosition]:
        """
        Get source position information if available.
        """
        return self.metadata.position if self.metadata else None


@dataclass
class Program(ASTNode):
    """
    Root node representing the entire program.
    """

    statements: List[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        """Initialize parent ASTNode after dataclass initialization."""
        super().__init__()

    def accept(self, visitor: Any) -> Any:
        """Accept visitor for program traversal."""
        return visitor.visit_program(self)


@dataclass
class Comment(ASTNode):
    text: str = ""

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_comment(self)


@dataclass
class Number(ASTNode):
    value: Union[int, float]

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_number(self)


@dataclass
class String(ASTNode):
    value: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_string(self)


@dataclass
class Var(ASTNode):
    name: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_var(self)


@dataclass
class Bool(ASTNode):
    value: bool

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_bool(self)


@dataclass
class Null(ASTNode):
    value: None = None

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_null(self)


@dataclass
class TupleLiteral(ASTNode):
    elements: List[Any]

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_tuple_literal(self)


@dataclass
class VarDecl(ASTNode):
    items: List[Any] = field(default_factory=list)

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_var_decl(self)


@dataclass
class Assignment(ASTNode):
    target: Any
    value: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_assignment(self)


@dataclass
class VarTarget(ASTNode):
    name: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_var_target(self)


@dataclass
class ArrayTarget(ASTNode):
    name: str | None
    index: List[Any]
    array: Any | None = None

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_array_target(self)


@dataclass
class FieldTarget(ASTNode):
    obj: str
    field: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_field_target(self)


@dataclass
class ForLoop(ASTNode):
    var: str
    start: Any
    end: Any
    step: Optional[Any] = None
    body: List[Any] = field(default_factory=List[Any])
    preserve_counter_value: bool = True

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_for_loop(self)


@dataclass
class ForEachLoop(ASTNode):
    var: str
    collection: Any
    body: List[Any] = field(default_factory=List[Any])
    preserve_counter_value: bool = True
    tuple_vars: Optional[List[str]] = None

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_for_each_loop(self)


@dataclass
class WhileLoop(ASTNode):
    cond: Any
    body: List[Any] = field(default_factory=List[Any])

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_while_loop(self)


@dataclass
class RepeatUntil(ASTNode):
    cond: Any
    body: List[Any] = field(default_factory=List[Any])

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_repeat_until(self)


@dataclass
class IfElse(ASTNode):
    cond: Any
    then_branch: List[Any] = field(default_factory=List[Any])
    else_branch: List[Any] = field(default_factory=List[Any])

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_if_else(self)


@dataclass
class CallStmt(ASTNode):
    name: str
    args: List[Any] = field(default_factory=List[Any])

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_call_stmt(self)


@dataclass
class CallMethod(ASTNode):
    obj: Any
    method: str
    args: List[Any] = field(default_factory=List[Any])

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_call_method(self)


@dataclass
class ReturnStmt(ASTNode):
    value: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_return_stmt(self)


@dataclass
class ArrayAccess(ASTNode):
    name: str | None
    index: List[Any]
    array: Any | None = None

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_array_access(self)


@dataclass
class ArraySlice(ASTNode):
    name: str
    ranges: List[Any]
    start: Optional[Any] = None
    end: Optional[Any] = None

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_array_slice(self)


@dataclass
class FieldAccess(ASTNode):
    obj: Any
    field: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_field_access(self)


@dataclass
class FuncCallExpr(ASTNode):
    name: str | Null
    args: List[Any] = field(default_factory=List[Any])

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_func_call_expr(self)


@dataclass
class BinOp(ASTNode):
    op: Any
    left: Any
    right: Any
    short_circuit: bool = False

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_bin_op(self)


@dataclass
class ShortCircuitBinOp(ASTNode):
    op: str
    left: Any
    right: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_short_circuit_bin_op(self)


@dataclass
class UnOp(ASTNode):
    op: str
    value: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_un_op(self)


@dataclass
class ClassDef(ASTNode):
    name: str | Null
    fields: List[str] = field(default_factory=List[str])

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_class_def(self)


@dataclass
class NewObject(ASTNode):
    class_name: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_new_object(self)


@dataclass
class Parameter(ASTNode):
    name: str | Null
    param_type: str
    dimensions: Optional[List[Any]]
    class_name: Optional[str]

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_parameter(self)


@dataclass
class SubroutineDef(ASTNode):
    name: str | Null
    parameters: List[Parameter] = field(default_factory=List[Parameter])
    body: List[ASTNode] = field(default_factory=List[ASTNode])

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_subroutine_def(self)


@dataclass
class LengthFunction(ASTNode):
    array: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_length_function(self)


@dataclass
class CeilFunction(ASTNode):
    expr: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_ceil_function(self)


@dataclass
class FloorFunction(ASTNode):
    expr: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_floor_function(self)


@dataclass
class StrlenFunction(ASTNode):
    expr: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_strlen_function(self)


@dataclass
class ConcatFunction(ASTNode):
    left: Any
    right: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_concat_function(self)


@dataclass
class SubstringFunction(ASTNode):
    string: Any
    start: Any
    length: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_substring_function(self)


@dataclass
class PrintStmt(ASTNode):
    value: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_print_stmt(self)


@dataclass
class AddNodeFunction(ASTNode):
    graph: str
    node: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_add_node_function(self)


@dataclass
class AddEdgeFunction(ASTNode):
    graph: str
    from_node: Any
    to_node: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_add_edge_function(self)


@dataclass
class NeighborsFunction(ASTNode):
    graph: str
    node: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_neighbors_function(self)


@dataclass
class ArrayVarDecl(ASTNode):
    name: str
    dimensions: List[Any]

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_array_var_decl(self)


@dataclass
class ObjectVarDecl(ASTNode):
    class_name: str
    name: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_object_var_decl(self)


@dataclass
class GraphVarDecl(ASTNode):
    name: str

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_graph_var_decl(self)


@dataclass
class GraphOperation(ASTNode):
    graph: str
    nodes: List[Any] = field(default_factory=List[Any])

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_graph_operation(self)


@dataclass
class GraphTraversal(ASTNode):
    graph: str
    start_node: Any
    end_node: Any

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_graph_traversal(self)


@dataclass
class NewGraph(ASTNode):

    def accept(self, visitor: Any) -> Any:
        return visitor.visit_new_graph(self)
