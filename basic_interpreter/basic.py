from contextlib import contextmanager
import re
from typing import Optional

from strings_with_arrows import string_with_arrows

#############################################
# Constants
#############################################
DIGITS = "0123456789"


#############################################
# Position
#############################################
class Position:
    def __init__(self, idx, line, col, filename, filetext):
        self.idx = idx
        self.line = line
        self.col = col
        self.filename = filename
        self.filetext = filetext

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == "\n":
            self.line += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.line, self.col, self.filename, self.filetext)


#############################################
# Errors
#############################################


class Error:
    def __init__(
        self, pos_start: Position, pos_end: Position, error_name: str, details: str
    ):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f"{self.error_name}: {self.details}\n"
        result += f"File: {self.pos_start.filename}, line: {self.pos_start.line + 1}"
        result += "\n\n" + string_with_arrows(
            self.pos_start.filetext, self.pos_start, self.pos_end
        )
        return result


class IllegalCharError(Error):
    def __init__(self, pos_start: Position, pos_end: Position, details: str):
        super().__init__(
            pos_start=pos_start,
            pos_end=pos_end,
            error_name="Illegal Character",
            details=details,
        )


class InvalidSyntaxError(Error):
    def __init__(self, pos_start: Position, pos_end: Position, details: str):
        super().__init__(
            pos_start=pos_start,
            pos_end=pos_end,
            error_name="Invalid Syntax",
            details=details,
        )


class RuntimeError(Error):
    def __init__(
        self, pos_start: Position, pos_end: Position, details: str, context: "Context"
    ):
        super().__init__(
            pos_start=pos_start,
            pos_end=pos_end,
            error_name="Runtime Error",
            details=details,
        )
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f"{self.error_name}: {self.details}\n"
        result += "\n\n" + string_with_arrows(
            self.pos_start.filetext, self.pos_start, self.pos_end
        )
        return result

    def generate_traceback(self):
        result = ""
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = (
                f"  File {pos.filename}, line {str(pos.line + 1)}, in {ctx.display_name}\n"
                + result
            )
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return "Traceback (most recent call last):\n" + result


#############################################
# Tokens
#############################################

TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_PLUS = "PLUS"
TT_MINUS = "MINUS"
TT_MUL = "MUL"
TT_DIV = "DIV"
TT_POW = "POW"
TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_EOF = "EOF"


class Token:
    def __init__(self, _type, value=None, pos_start=None, pos_end=None):
        self.type = _type
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()

    def __repr__(self):
        return f"{self.type}:{self.value}" if self.value else f"{self.type}"


#############################################
# Lexer
#############################################


class Lexer:
    def __init__(self, filename, text):
        self.filename = filename
        self.text = text
        self.pos = Position(-1, 0, -1, self.filename, self.text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = (
            self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
        )

    def make_tokens(self):
        tokens = []

        while self.current_char is not None:
            if self.current_char in " \t":
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
                # self.advance()
            elif self.current_char == "+":
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == "-":
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == "*":
                # tokens.append(Token(TT_MUL, pos_start=self.pos))
                # self.advance()
                prev_pos = self.pos
                self.advance()
                if self.current_char == "*":
                    tokens.append(Token(TT_POW, pos_start=prev_pos))
                    self.advance()
                else:
                    tokens.append(Token(TT_MUL, pos_start=prev_pos))

            elif self.current_char == "/":
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == "(":
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ")":
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, f"'{char}'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ""
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in DIGITS + ".":
            if self.current_char == ".":
                if dot_count == 1:
                    break

                dot_count += 1
                num_str += "."
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            # is an integer type
            return Token(TT_INT, int(num_str), pos_start=pos_start, pos_end=self.pos)
        else:
            # is a float type
            return Token(
                TT_FLOAT, float(num_str), pos_start=pos_start, pos_end=self.pos
            )


#############################################
# NODES
#############################################


class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = tok.pos_start
        self.pos_end = tok.pos_end

    def __repr__(self) -> str:
        return f"{self.tok}"


class BinOpNode:

    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self) -> str:
        return f"({self.left_node}, {self.op_tok}, {self.right_node})"


class UnaryOpNode:

    def __init__(self, op_tok, node) -> None:
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = self.node.pos_end

    def __repr__(self) -> str:
        return f"({self.op_tok}, {self.node})"


#############################################
# PARSER RESULT
#############################################


class ParseResult:

    def __init__(self) -> None:
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error:
                self.error = res.error

            return res.node

        return res

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


#############################################
# PARSER
#############################################


class Parser:

    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        # self.current_tok: Optional[Token] = None
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    #############################################

    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(
                InvalidSyntaxError(
                    pos_start=self.current_tok.pos_start,
                    pos_end=self.current_tok.pos_end,
                    details="Expected '+', '-', '*' or '/'",
                )
            )

        return res

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error:
                return res

            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(
                    InvalidSyntaxError(
                        pos_start=self.current_tok.pos_start,
                        pos_end=self.current_tok.pos_end,
                        details="Expected ')'",
                    )
                )

        return res.failure(
            InvalidSyntaxError(
                pos_start=tok.pos_start,
                pos_end=tok.pos_end,
                details="Expected int, float, '+', '-' or '('",
            )
        )

    def power(self):
        return self.bin_op(
            atomic_func=self.atom, ops=(TT_POW,), factor_func=self.factor
        )

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error:
                return res

            return res.success(UnaryOpNode(op_tok=tok, node=factor))

        return self.power()

    def term(self):
        return self.bin_op(atomic_func=self.factor, ops=(TT_MUL, TT_DIV, TT_POW))

    def expr(self):
        return self.bin_op(atomic_func=self.term, ops=(TT_PLUS, TT_MINUS))

    #############################################

    def bin_op(self, atomic_func, ops, factor_func=None):
        if factor_func is None:
            factor_func = atomic_func

        res = ParseResult()
        left = res.register(atomic_func())
        if res.error:
            return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(factor_func())

            if res.error:
                return res

            left = BinOpNode(left_node=left, op_tok=op_tok, right_node=right)

        return res.success(left)


#############################################
# RUNTIME RESULT
#############################################


class RTResult:

    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res: "RTResult"):
        if res.error:
            self.error = res.error

        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


#############################################
# VALUES
#############################################


class Number:
    def __init__(self, value):
        self.value = value
        self.set_context()
        self.set_pos()

    def set_context(self, context: "Context | None" = None):
        self.context = context
        return self

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RuntimeError(
                    pos_start=other.pos_start,
                    pos_end=other.pos_end,
                    details="Division by zero",
                    context=self.context,
                )
            return Number(self.value / other.value).set_context(self.context), None

    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value**other.value).set_context(self.context), None

    def __repr__(self) -> str:
        return str(self.value)


#############################################
# CONTEXT
#############################################


class Context:

    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos


#############################################
# INTERPRETER
#############################################


class Interpreter:

    def visit(self, node, context: Context):
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f"No visit_{type(node).__name__} method defined")

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value)
            .set_context(context)
            .set_pos(pos_start=node.pos_start, pos_end=node.pos_end)
        )

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        result, error = None, None
        op_tok = node.op_tok
        if op_tok.type == TT_PLUS:
            result, error = left.added_to(right)
        if op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        if op_tok.type == TT_MUL:
            result, error = left.multed_by(right)
        if op_tok.type == TT_DIV:
            result, error = left.dived_by(right)
        if op_tok.type == TT_POW:
            result, error = left.powed_by(right)

        if error:
            return res.failure(error=error)

        result.set_pos(pos_start=node.pos_start, pos_end=node.pos_end)
        return res.success(result)

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error:
            return res

        error = None
        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_by(Number(-1))

        if error:
            return res.failure(error)

        return res.success(
            number.set_pos(pos_start=node.pos_start, pos_end=node.pos_end)
        )


#############################################
# RUN
#############################################

#  TODO: Continuar Aula 04 - começo do vídeo


def run(filename: str, text: str):
    # Generate tokens
    lexer = Lexer(filename, text)
    tokens, error = lexer.make_tokens()

    if error:
        return None, error

    # Generate AST
    parser = Parser(tokens=tokens)
    ast = parser.parse()

    if ast.error:
        return None, ast.error

    # Run program
    interpreter = Interpreter()
    context = Context("<program>")

    result = interpreter.visit(ast.node, context)

    return result.value, result.error
