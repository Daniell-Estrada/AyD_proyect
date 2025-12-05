"""
Lark Transformer implementation to convert parse trees into AST nodes.
Handles all grammar constructs including loops, conditionals, recursion, arrays, objects, and graphs.
"""

from typing import Any, List, Optional, Union

from lark import Token, Transformer

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
    ConcatFunction,
    FieldAccess,
    FieldTarget,
    FloorFunction,
    ForLoop,
    ForEachLoop,
    FuncCallExpr,
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
    String,
    StrlenFunction,
    SubroutineDef,
    SubstringFunction,
    TupleLiteral,
    UnOp,
    Var,
    VarDecl,
    VarTarget,
    WhileLoop,
)


class ASTTransformer(Transformer):
    """
    Transforms Lark parse tree into custom AST nodes.
    Each method corresponds to a rule in grammar.lark. Lark invokes these
    handlers via reflection, so they may appear unused to static analyzers.
    """

    # ===== Internal Utilities =====

    def _assign_position(self, node: Any, token: Token) -> Any:
        """Attach source position metadata to an AST node when available."""

        if not hasattr(token, "line"):
            return node

        line = getattr(token, "line", 0)
        column = getattr(token, "column", 0)
        end_line = getattr(token, "end_line", line)
        end_column = getattr(token, "end_column", column)

        if hasattr(node, "set_position"):
            node.set_position(
                line=line,
                column=column,
                end_line=end_line,
                end_column=end_column,
            )

        return node

    # ===== Program Structure =====

    def program(self, items: List[Any]) -> Program:
        """Transform program root node."""
        statements = [item for item in items if item is not None]
        return Program(statements=statements)

    def start(self, items: List[Any]) -> Program:
        """Transform start rule."""
        return items[0] if items else Program(statements=[])

    # ===== Literals =====

    def number(self, items: List[Token]) -> Number:
        """Transform number literal."""
        value_str = items[0].value
        value = float(value_str) if "." in value_str else int(value_str)
        return self._assign_position(Number(value=value), items[0])

    def string(self, items: List[Token]) -> String:
        """Transform string literal."""
        # Remove quotes from string
        value = items[0].value[1:-1]
        return self._assign_position(String(value=value), items[0])

    def true(self, items: List[Any]) -> Bool:
        """Transform boolean true."""
        return self._assign_position(Bool(value=True), items[0])

    def false(self, items: List[Any]) -> Bool:
        """Transform boolean false."""
        return self._assign_position(Bool(value=False), items[0])

    def null(self, items: List[Any]) -> Null:
        """Transform null literal."""
        token = items[0] if items else None
        node = Null()
        return self._assign_position(node, token) if token else node

    def var(self, items: List[Token]) -> Var:
        """Transform variable reference."""
        return self._assign_position(Var(name=items[0].value), items[0])

    # ===== Declarations =====

    def var_decl(self, items: List[Any]) -> VarDecl:
        """Transform variable declaration."""
        token = next((i for i in items if isinstance(i, Token)), None)
        node = VarDecl(items=items)
        return self._assign_position(node, token) if token else node

    def array_var_decl(self, items: List[Any]) -> ArrayVarDecl:
        """Transform array variable declaration."""
        name = items[0].value
        dimensions = items[1:]
        return self._assign_position(ArrayVarDecl(name=name, dimensions=dimensions), items[0])

    def object_var_decl(self, items: List[Any]) -> ObjectVarDecl:
        """Transform object variable declaration."""
        class_name = items[0].value
        name = items[1].value
        return self._assign_position(ObjectVarDecl(class_name=class_name, name=name), items[0])

    def graph_var_decl(self, items: List[Token]) -> GraphVarDecl:
        """Transform graph variable declaration."""
        name = items[0].value
        return self._assign_position(GraphVarDecl(name=name), items[0])

    # ===== Assignments =====

    def assignment(self, items: List[Any]) -> Assignment:
        """Transform assignment statement."""
        meaningful = [item for item in items if not isinstance(item, Token)]
        target = meaningful[0]
        value = meaningful[1] if len(meaningful) > 1 else None
        token = next((i for i in items if isinstance(i, Token)), None)
        node = Assignment(target=target, value=value)
        return self._assign_position(node, token) if token else node

    def lvalue_var(self, items: List[Token]) -> VarTarget:
        """Transform simple variable lvalue."""
        return self._assign_position(VarTarget(name=items[0].value), items[0])

    def lvalue_array(self, items: List[Any]) -> ArrayTarget:
        """Transform array element lvalue."""
        name = items[0].value
        indices = items[1:]
        return self._assign_position(ArrayTarget(name=name, index=indices), items[0])

    def lvalue_field(self, items: List[Token]) -> FieldTarget:
        """Transform field access lvalue."""
        obj = items[0].value
        field = items[1].value
        return self._assign_position(FieldTarget(obj=obj, field=field), items[0])

    # ===== Control Flow =====

    def for_loop(self, items: List[Any]) -> ForLoop:
        """Transform for loop."""
        meaning = [
            item
            for item in items
            if not isinstance(item, Token) or item.type == "VAR"
        ]

        var_token = meaning[0]
        start = meaning[1]
        end = meaning[2]
        body_items = meaning[3]
        body = body_items.statements if isinstance(body_items, Program) else [body_items]
        return self._assign_position(
            ForLoop(var=var_token.value, start=start, end=end, body=body),
            var_token if isinstance(var_token, Token) else None,
        )

    def for_each_loop_assign(self, items: List[Any]) -> ForEachLoop:
        """Transform for-each loop using assignment syntax."""
        meaning = [
            item
            for item in items
            if not isinstance(item, Token) or item.type == "VAR"
        ]

        if len(meaning) == 4:
            var_token = meaning[1]
            collection = meaning[2]
            body_items = meaning[3]
        else:
            var_token = meaning[0]
            collection = meaning[1]
            body_items = meaning[2]
        body = body_items.statements if isinstance(body_items, Program) else [body_items]
        return self._assign_position(
            ForEachLoop(var=var_token.value, collection=collection, body=body),
            var_token if isinstance(var_token, Token) else None,
        )

    def for_each_loop_in(self, items: List[Any]) -> ForEachLoop:
        """Transform for-each loop using 'in' syntax."""
        meaning = [
            item
            for item in items
            if not isinstance(item, Token) or item.type == "VAR"
        ]

        if len(meaning) == 4:
            var_token = meaning[1]
            collection = meaning[2]
            body_items = meaning[3]
        else:
            var_token = meaning[0]
            collection = meaning[1]
            body_items = meaning[2]
        body = body_items.statements if isinstance(body_items, Program) else [body_items]
        return self._assign_position(
            ForEachLoop(var=var_token.value, collection=collection, body=body),
            var_token if isinstance(var_token, Token) else None,
        )

    def while_loop(self, items: List[Any]) -> WhileLoop:
        """Transform while loop."""
        meaningful = [
            item
            for item in items
            if not isinstance(item, Token) or item.type not in {"WHILE", "DO"}
        ]

        condition = meaningful[0]
        body_items = meaningful[1]
        body = body_items.statements if isinstance(body_items, Program) else [body_items]
        token = next((t for t in items if isinstance(t, Token)), None)
        return self._assign_position(WhileLoop(cond=condition, body=body), token) if token else WhileLoop(cond=condition, body=body)

    def repeat_loop(self, items: List[Any]) -> RepeatUntil:
        """Transform repeat-until loop."""
        # Body statements come before condition
        condition = items[-1]
        body_items = items[:-1]
        statements: List[Any] = []
        for item in body_items:
            if isinstance(item, Program):
                statements.extend(item.statements)
            elif item is not None and not isinstance(item, Token):
                statements.append(item)

        token = next((t for t in items if isinstance(t, Token)), None)
        return self._assign_position(RepeatUntil(cond=condition, body=statements), token) if token else RepeatUntil(cond=condition, body=statements)

    def if_statement(self, items: List[Any]) -> IfElse:
        """Transform if-then-else statement."""
        filtered = [item for item in items if not isinstance(item, Token)]
        condition = filtered[0]
        then_branch = filtered[1]
        else_branch = filtered[2] if len(filtered) > 2 else Program(statements=[])

        then_body = (
            then_branch.statements if isinstance(then_branch, Program) else [then_branch]
        )
        else_body = (
            else_branch.statements if isinstance(else_branch, Program) else [else_branch]
        )

        token = next((t for t in items if isinstance(t, Token)), None)
        return self._assign_position(
            IfElse(cond=condition, then_branch=then_body, else_branch=else_body),
            token,
        ) if token else IfElse(cond=condition, then_branch=then_body, else_branch=else_body)

    def code_block(self, items: List[Any]) -> Program:
        """Transform code block (begin...end)."""
        statements = []
        for item in items:
            if isinstance(item, Program):
                statements.extend(item.statements)
            elif item is not None and not isinstance(item, Token):
                statements.append(item)
        return Program(statements=statements)

    def statement_block(self, items: List[Any]) -> Program:
        """Transform newline-delimited blocks terminated by END."""
        statements = [
            item for item in items
            if item is not None and not isinstance(item, Token)
        ]
        return Program(statements=statements)

    def suite(self, items: List[Any]) -> Program:
        """Transform a generic statement suite without explicit delimiters."""
        statements = [
            item for item in items
            if item is not None and not isinstance(item, Token)
        ]
        return Program(statements=statements)

    # ===== Functions and Calls =====

    def subroutine_def(self, items: List[Any]) -> SubroutineDef:
        """Transform function definition (handles optional FUNCTION token)."""
        idx = 0
        if items and isinstance(items[0], Token) and items[0].type == "FUNCTION":
            idx += 1

        name_token = items[idx] if idx < len(items) else None
        name = name_token.value if isinstance(name_token, Token) else None
        idx += 1

        parameters: List[Parameter] = []
        body: Program | None = None

        for item in items[idx:]:
            if isinstance(item, Parameter):
                parameters.append(item)
            elif isinstance(item, Program):
                body = item
                break

        body_statements = body.statements if body else []
        token = name_token if isinstance(name_token, Token) else None
        return self._assign_position(
            SubroutineDef(name=name, parameters=parameters, body=body_statements),
            token,
        ) if token else SubroutineDef(name=name, parameters=parameters, body=body_statements)

    def simple_parameter(self, items: List[Token]) -> Parameter:
        """Transform simple parameter."""
        name = items[0].value
        return self._assign_position(
            Parameter(name=name, param_type="simple", dimensions=None, class_name=None),
            items[0],
        )

    def array_parameter(self, items: List[Any]) -> Parameter:
        """Transform array parameter."""
        name = items[0].value
        dimensions = items[1:]
        return self._assign_position(
            Parameter(name=name, param_type="array", dimensions=dimensions, class_name=None),
            items[0],
        )

    def object_parameter(self, items: List[Token]) -> Parameter:
        """Transform object parameter."""
        class_name = items[0].value
        name = items[1].value
        return self._assign_position(
            Parameter(name=name, param_type="object", dimensions=None, class_name=class_name),
            items[0],
        )

    def graph_parameter(self, items: List[Token]) -> Parameter:
        """Transform graph parameter."""
        name = items[0].value
        return self._assign_position(
            Parameter(name=name, param_type="graph", dimensions=None, class_name=None),
            items[0],
        )

    def func_call_expr(self, items: List[Any]) -> FuncCallExpr:
        """Transform function call expression."""
        name = items[0].value if hasattr(items[0], "value") else str(items[0])
        args = items[1:] if len(items) > 1 else []
        token = items[0] if isinstance(items[0], Token) else None
        return self._assign_position(FuncCallExpr(name=name, args=args), token) if token else FuncCallExpr(name=name, args=args)

    def call_stmt(self, items: List[Any]) -> CallStmt:
        """Transform call statement."""
        meaningful = [
            item
            for item in items
            if not isinstance(item, Token) or item.type in {"VAR", "ADDNODE", "ADDEDGE"}
        ]

        name_token = meaningful[0]
        args = [arg for arg in meaningful[1:]]
        return self._assign_position(CallStmt(name=name_token.value, args=args), name_token if isinstance(name_token, Token) else None)

    def call_method(self, items: List[Any]) -> CallMethod:
        """Transform method call."""
        obj = items[0]
        method = items[1].value
        args = items[2:] if len(items) > 2 else []
        return self._assign_position(CallMethod(obj=obj, method=method, args=args), items[1] if isinstance(items[1], Token) else None)

    def method_call_base(self, items: List[Any]) -> CallMethod:
        """Transform base method call like obj.method(...)."""
        obj_token = items[0]
        method_token = items[1]
        args = items[2:] if len(items) > 2 else []
        obj = Var(name=obj_token.value) if isinstance(obj_token, Token) else obj_token
        return self._assign_position(CallMethod(obj=obj, method=method_token.value, args=args), method_token if isinstance(method_token, Token) else None)

    def return_stmt(self, items: List[Any]) -> ReturnStmt:
        """Transform return statement."""
        meaningful_items = [item for item in items if not isinstance(item, Token)]
        value = meaningful_items[0] if meaningful_items else None
        token = next((t for t in items if isinstance(t, Token)), None)
        return self._assign_position(ReturnStmt(value=value), token) if token else ReturnStmt(value=value)

    def expression_list(self, items: List[Any]) -> TupleLiteral:
        """Transform expression list a, b, c as tuple literal."""
        return TupleLiteral(elements=items)

    # ===== Binary Operations =====

    def add(self, items: List[Any]) -> BinOp:
        """Transform addition."""
        left, right = self._resolve_binary_operands(items)
        return BinOp(op="+", left=left, right=right)

    def sub(self, items: List[Any]) -> BinOp:
        """Transform subtraction."""
        left, right = self._resolve_binary_operands(items)
        return BinOp(op="-", left=left, right=right)

    def mul(self, items: List[Any]) -> BinOp:
        """Transform multiplication."""
        left, right = self._resolve_binary_operands(items)
        return BinOp(op="*", left=left, right=right)

    def div(self, items: List[Any]) -> BinOp:
        """Transform division."""
        left, right = self._resolve_binary_operands(items)
        return BinOp(op="/", left=left, right=right)

    def mod(self, items: List[Any]) -> BinOp:
        """Transform modulo."""
        left, right = self._resolve_binary_operands(items)
        return BinOp(op="mod", left=left, right=right)

    def divint(self, items: List[Any]) -> BinOp:
        """Transform integer division."""
        left, right = self._resolve_binary_operands(items)
        return BinOp(op="div", left=left, right=right)

    def _resolve_binary_operands(self, items: List[Any]) -> tuple[Any, Any]:
        """Extract operands from production items, ignoring operator tokens."""
        operands = [item for item in items if not isinstance(item, Token)]
        if len(operands) != 2:
            raise ValueError(f"Expected 2 operands, got {len(operands)} from {items}")
        return operands[0], operands[1]

    def or_expr(self, items: List[Any]) -> Union[Any, BinOp]:
        """Transform OR expression."""
        if len(items) == 1:
            return items[0]
        result = items[0]
        for item in items[1:]:
            result = BinOp(op="or", left=result, right=item, short_circuit=True)
        return result

    def and_expr(self, items: List[Any]) -> Union[Any, BinOp]:
        """Transform AND expression."""
        if len(items) == 1:
            return items[0]
        result = items[0]
        for item in items[1:]:
            result = BinOp(op="and", left=result, right=item, short_circuit=True)
        return result

    def not_expr(self, items: List[Any]) -> Union[Any, UnOp]:
        """Transform NOT expression."""
        if len(items) == 1:
            return items[0]
        # NOT operator
        return UnOp(op="not", value=items[1])

    def comp_expr(self, items: List[Any]) -> Union[Any, BinOp]:
        """Transform comparison expression."""
        if len(items) == 1:
            return items[0]
        # Comparison operator
        left = items[0]
        op = items[1].value
        right = items[2]
        return BinOp(op=op, left=left, right=right)

    # ===== Array and Object Access =====

    def array_access(self, items: List[Any]) -> ArrayAccess:
        """Transform array element access."""
        name = items[0].value
        indices = items[1:]
        return ArrayAccess(name=name, index=indices)

    def array_slice(self, items: List[Any]) -> ArraySlice:
        """Transform array slice."""
        name = items[0].value
        start = items[1] if len(items) > 1 else None
        end = items[2] if len(items) > 2 else None
        return ArraySlice(name=name, ranges=items[1:], start=start, end=end)

    def field_access(self, items: List[Token]) -> FieldAccess:
        """Transform chained object field access (a.b.c → obj=a, field='b.c')."""
        obj = items[0].value
        field = ".".join(token.value for token in items[1:])
        return FieldAccess(obj=obj, field=field)

    # ===== Indexers =====

    def indexer_single(self, items: List[Any]) -> Any:
        """Transform single index."""
        return items[0]

    def indexer_range(self, items: List[Any]) -> dict:
        """Transform range indexer."""
        # Items structure: [start_expr, Token('RANGE', '..'), end_expr]
        return {"type": "range", "start": items[0], "end": items[2]}

    def indexer_open_start(self, items: List[Any]) -> dict:
        """Transform open start range."""
        # Items structure: [Token('RANGE', '..'), end_expr]
        return {"type": "range", "start": None, "end": items[1]}

    def indexer_open_end(self, items: List[Any]) -> dict:
        """Transform open end range."""
        return {"type": "range", "start": items[0], "end": None}

    def indexer_open_both(self, items: List[Any]) -> dict:
        """Transform fully open range."""
        return {"type": "range", "start": None, "end": None}

    def indexer_empty(self, items: List[Any]) -> dict:
        """Transform empty indexer."""
        return {"type": "empty"}

    def for_each_tuple_in(self, items: List[Any]) -> Any:
        """Transform tuple destructuring in for-each loops (for each (u, v) in collection)."""
        # items: [first_var, second_var, collection, body]
        first_var, second_var, collection, body_items = items
        body = body_items.statements if isinstance(body_items, Program) else [body_items]

        loop = ForEachLoop(var="tuple", collection=collection, body=body)
        loop.tuple_vars = [  # type: ignore[attr-defined]
            first_var.value if hasattr(first_var, "value") else str(first_var),
            second_var.value if hasattr(second_var, "value") else str(second_var),
        ]
        return loop

    def for_each_named_tuple_in(self, items: List[Any]) -> Any:
        """Transform named foreach tuple (e.g., for each edge (u, v) in G.edges)."""
        meaningful = [item for item in items if not isinstance(item, Token) or item.type == "VAR"]

        descriptor = meaningful[0]
        first_var = meaningful[1]
        second_var = meaningful[2]
        collection = meaningful[3]
        body_items = meaningful[4]
        body = body_items.statements if isinstance(body_items, Program) else [body_items]

        loop = ForEachLoop(var=descriptor.value, collection=collection, body=body)
        loop.tuple_vars = [  # type: ignore[attr-defined]
            first_var.value if hasattr(first_var, "value") else str(first_var),
            second_var.value if hasattr(second_var, "value") else str(second_var),
        ]
        return loop

    # ===== Built-in Functions =====

    def length_function(self, items: List[Any]) -> LengthFunction:
        """Transform length function."""
        return LengthFunction(array=items[0])

    def ceil_function(self, items: List[Any]) -> CeilFunction:
        """Transform ceil function."""
        return CeilFunction(expr=items[0])

    def floor_function(self, items: List[Any]) -> FloorFunction:
        """Transform floor function."""
        return FloorFunction(expr=items[0])

    def strlen_function(self, items: List[Any]) -> StrlenFunction:
        """Transform strlen function."""
        return StrlenFunction(expr=items[0])

    def concat_function(self, items: List[Any]) -> ConcatFunction:
        """Transform concat function."""
        return ConcatFunction(left=items[0], right=items[1])

    def substring_function(self, items: List[Any]) -> SubstringFunction:
        """Transform substring function."""
        return SubstringFunction(string=items[0], start=items[1], length=items[2])

    def tuple_expr(self, items: List[Any]) -> TupleLiteral:
        """Transform tuple literal (expr, expr, ...)."""
        return TupleLiteral(elements=items)

    def print_stmt(self, items: List[Any]) -> PrintStmt:
        """Transform print statement."""
        value = next((item for item in items if not isinstance(item, Token)), None)
        return PrintStmt(value=value)

    def print_function(self, items: List[Any]) -> PrintStmt:
        """Transform print function (in expression context)."""
        value = next((item for item in items if not isinstance(item, Token)), None)
        return PrintStmt(value=value)

    # ===== Graph Operations =====

    def addnode_function(self, items: List[Any]) -> AddNodeFunction:
        """Transform addNode function."""
        graph = items[0].value
        node = items[1]
        return AddNodeFunction(graph=graph, node=node)

    def addedge_function(self, items: List[Any]) -> AddEdgeFunction:
        """Transform addEdge function."""
        graph = items[0].value
        from_node = items[1]
        to_node = items[2]
        return AddEdgeFunction(graph=graph, from_node=from_node, to_node=to_node)

    def neighbors_function(self, items: List[Any]) -> NeighborsFunction:
        """Transform neighbors function."""
        meaningful = [item for item in items if not (isinstance(item, Token) and item.type == "NEIGHBORS")]

        if not meaningful:
            return NeighborsFunction(graph="", node=None)

        graph_item = meaningful[0]
        node = meaningful[1] if len(meaningful) > 1 else None

        graph_name = graph_item.value if isinstance(graph_item, Token) else getattr(graph_item, "name", str(graph_item))

        if isinstance(node, Token):
            node = Var(name=node.value)

        return NeighborsFunction(graph=graph_name, node=node)

    def addnode_stmt(self, items: List[Any]) -> AddNodeFunction:
        """Transform addNode statement."""
        graph = items[0].value
        node = items[1]
        return AddNodeFunction(graph=graph, node=node)

    def addedge_stmt(self, items: List[Any]) -> AddEdgeFunction:
        """Transform addEdge statement."""
        graph = items[0].value
        from_node = items[1]
        to_node = items[2]
        return AddEdgeFunction(graph=graph, from_node=from_node, to_node=to_node)

    # ===== Class and Object =====

    def class_def(self, items: List[Any]) -> ClassDef:
        """Transform class definition."""
        name = items[0].value if items[0] else None
        fields = [item.value for item in items[1:] if hasattr(item, "value")]
        return ClassDef(name=name, fields=fields)

    def new_object(self, items: List[Token]) -> NewObject:
        """Transform new object instantiation."""
        class_name = items[0].value
        return NewObject(class_name=class_name)

    def new_graph(self, items: List[Any]) -> NewGraph:
        """Transform new graph instantiation."""
        return NewGraph()
