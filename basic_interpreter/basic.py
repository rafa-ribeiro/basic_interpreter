#############################################
# Constants
#############################################
DIGITS = '0123456789'

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

    def advance(self, current_char):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.line += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.line, self.col, self.filename, self.filetext)

#############################################
# Errors
#############################################

class Error:

    def __init__(self, pos_start: int, pos_end: int, error_name: str, details: str):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'File: {self.pos_start.filename}, line: {self.pos_start.line + 1}'
        return result

class IllegalCharError(Error):

    def __init__(self, pos_start: Position, pos_end: Position, details: str):
        super().__init__(
                pos_start=pos_start, 
                pos_end=pos_end, 
                error_name='Illegal Character', 
                details=details
            )


#############################################
# Tokens
#############################################

TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'

class Token:
    
    def __init__(self, _type, value=None):
        self.type = _type
        self.value = value

    def __repr__(self):
        return f'{self.type}:{self.value}' if self.value else f'{self.type}'


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
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []
        
        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
                # self.advance()
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char 
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, f"'{char}'")

        return tokens, None


    def make_number(self):
        num_str = ''
        dot_count = 0

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count ==  1: 
                    break
                
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            # is an integer type
            return Token(TT_INT, int(num_str))
        else:
            # is a float type
            return Token(TT_FLOAT, float(num_str))


#############################################
# RUN
#############################################

def run(filename: str, text: str):
    lexer = Lexer(filename, text)
    tokens, error = lexer.make_tokens()

    return tokens, error



