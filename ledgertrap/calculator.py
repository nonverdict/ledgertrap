"""Arithmetic-only calculator: decimals, + - * /, parentheses, unary minus. No eval, no names."""

import re
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

MAX_LEN = 200
MAX_DEPTH = 50
_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d*)?|\.\d+)|([+\-*/()]))")


class CalcError(ValueError):
    pass


def _tokenize(expr):
    pos, out = 0, []
    expr = expr.rstrip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            raise CalcError("bad token")
        out.append(("num", m.group(1)) if m.group(1) is not None else ("op", m.group(2)))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0
        self.depth = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self):
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self):
        v = self.factor()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            r = self.factor()
            if op == "*":
                v = v * r
            else:
                if r == 0:
                    raise CalcError("division by zero")
                v = v / r
        return v

    def factor(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalcError("too deep")
        try:
            kind, val = self.take()
            if kind == "num":
                return Decimal(val)
            if (kind, val) == ("op", "-"):
                return -self.factor()
            if (kind, val) == ("op", "+"):
                return self.factor()
            if (kind, val) == ("op", "("):
                v = self.expr()
                if self.take() != ("op", ")"):
                    raise CalcError("missing )")
                return v
            raise CalcError("unexpected token")
        finally:
            self.depth -= 1


def _fmt(v: Decimal) -> str:
    if v == v.to_integral_value():
        return str(int(v))
    s = format(v.normalize(), "f")
    return s


def evaluate(expression) -> str:
    if not isinstance(expression, str):
        raise CalcError("expression must be a string")
    if len(expression) > MAX_LEN or not expression.strip():
        raise CalcError("bad length")
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.traps[DivisionByZero] = True
        ctx.traps[InvalidOperation] = True
        try:
            p = _Parser(_tokenize(expression))
            v = p.expr()
            if p.i != len(p.t):
                raise CalcError("trailing tokens")
            return _fmt(v)
        except (InvalidOperation, DivisionByZero, ArithmeticError) as e:
            raise CalcError(str(type(e).__name__)) from None
