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


#############################################
# Tokens
#############################################

TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_PLUS = "PLUS"
TT_MINUS = "MINUS"
TT_MUL = "MUL"
TT_DIV = "DIV"
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
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
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

    def __repr__(self) -> str:
        return f"{self.tok}"


class BinOpNode:

    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

    def __repr__(self) -> str:
        return f"({self.left_node}, {self.op_tok}, {self.right_node})"


class UnaryOpNode:

    def __init__(self, op_tok, node) -> None:
        self.op_tok = op_tok
        self.node = node

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

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error:
                return res

            return res.success(UnaryOpNode(op_tok=tok, node=factor))

        elif tok.type in (TT_INT, TT_FLOAT):
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
                details="Expected int or float",
            )
        )

    def term(self):
        return self.bin_op(func=self.factor, ops=(TT_MUL, TT_DIV))

    def expr(self):
        return self.bin_op(func=self.term, ops=(TT_PLUS, TT_MINUS))

    #############################################

    def bin_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error:
            return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())

            if res.error:
                return res

            left = BinOpNode(left_node=left, op_tok=op_tok, right_node=right)

        return res.success(left)


#############################################
# RUN
#############################################

#  TODO: Continuar no minuto 9:55 do ep. 2


def run(filename: str, text: str):
    # Generate tokens
    lexer = Lexer(filename, text)
    tokens, error = lexer.make_tokens()

    if error:
        return None, error

    # Generate AST
    parser = Parser(tokens=tokens)
    ast = parser.parse()

    return ast.node, ast.error
