/**
 * Deterministic recompute pass.
 *
 * Detects arithmetic expressions, `range(...)` enumerations, leap-year
 * assertions, and a small set of unit-arithmetic claims in the assistant's
 * answer, and confirms them by actually computing the result. When the
 * computed value disagrees with the answer's claim, the aggregator treats
 * it as a hard fail (no LLM uncertainty involved). When the computed value
 * agrees, the aggregator can *suppress* an NLI contradiction flag on the
 * same expression — which targets the particular failure mode where the
 * LLM claim-checker false-flags correct arithmetic.
 *
 * Scope is deliberately narrow. The parser handles:
 *   - integer / decimal numbers (English thousands separators tolerated)
 *   - +  -  *  /  %  ^  (with correct precedence and right-assoc ^)
 *   - unary minus
 *   - parentheses
 *   - implicit multiplication (`3(5)`, `(2)(3)`, `2(x+1)` with x resolved)
 *
 * Things explicitly out of scope for v1 (route to critic panel):
 *   - variables / substitution
 *   - functions beyond factorial (`5! = 120`)
 *   - transcendentals (sin, cos, log, sqrt)
 *   - full Python list-comprehension evaluation (only three patterns)
 *   - symbolic algebra
 */

import type { RecomputeResult, RecomputeVerification } from "../types.js";

// ───────────────────────────────────────────────────────────────────────────
// Tokeniser + parser for arithmetic expressions (safe — no eval)
// ───────────────────────────────────────────────────────────────────────────

type Token =
  | { t: "NUM"; v: number }
  | { t: "OP"; v: "+" | "-" | "*" | "/" | "%" | "^" }
  | { t: "LP" }
  | { t: "RP" };

/** Tokenise a cleaned arithmetic string. Returns null on any unexpected char. */
function tokenise(s: string): Token[] | null {
  const tokens: Token[] = [];
  const n = s.length;
  let i = 0;
  while (i < n) {
    const c = s[i]!;
    if (c === " " || c === "\t") {
      i++;
      continue;
    }
    // Number: leading digit or decimal point
    if ((c >= "0" && c <= "9") || c === ".") {
      let j = i;
      // integer part (allow commas as thousands separators)
      while (j < n && ((s[j]! >= "0" && s[j]! <= "9") || s[j] === ",")) j++;
      // fractional part
      if (j < n && s[j] === ".") {
        j++;
        while (j < n && s[j]! >= "0" && s[j]! <= "9") j++;
      }
      const raw = s.slice(i, j).replace(/,/g, "");
      const v = Number(raw);
      if (!Number.isFinite(v)) return null;
      tokens.push({ t: "NUM", v });
      i = j;
      continue;
    }
    if (c === "+" || c === "-" || c === "*" || c === "/" || c === "%" || c === "^") {
      tokens.push({ t: "OP", v: c });
      i++;
      continue;
    }
    if (c === "(") {
      tokens.push({ t: "LP" });
      i++;
      continue;
    }
    if (c === ")") {
      tokens.push({ t: "RP" });
      i++;
      continue;
    }
    // Unexpected character — give up rather than guess.
    return null;
  }
  return tokens;
}

/**
 * Insert implicit-multiplication tokens.
 * Wherever NUM or RP is immediately followed by NUM or LP, insert '*'.
 */
function insertImplicitMul(tokens: Token[]): Token[] {
  const out: Token[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]!;
    out.push(t);
    if (i + 1 < tokens.length) {
      const next = tokens[i + 1]!;
      const needsMul =
        (t.t === "NUM" || t.t === "RP") && (next.t === "NUM" || next.t === "LP");
      if (needsMul) {
        out.push({ t: "OP", v: "*" });
      }
    }
  }
  return out;
}

/** Recursive-descent parser over tokens. Returns computed value or null. */
function evalTokens(tokens: Token[]): number | null {
  const state = { i: 0 };

  const peek = (): Token | null =>
    state.i < tokens.length ? tokens[state.i]! : null;

  const eat = (): Token | null =>
    state.i < tokens.length ? tokens[state.i++]! : null;

  // expr = term (('+' | '-') term)*
  const parseExpr = (): number | null => {
    let left = parseTerm();
    if (left === null) return null;
    // biome-ignore lint: intentional infinite loop with break
    while (true) {
      const p = peek();
      if (p && p.t === "OP" && (p.v === "+" || p.v === "-")) {
        const op = p.v;
        eat();
        const right = parseTerm();
        if (right === null) return null;
        left = op === "+" ? left + right : left - right;
      } else break;
    }
    return left;
  };

  // term = factor (('*' | '/' | '%') factor)*
  const parseTerm = (): number | null => {
    let left = parseFactor();
    if (left === null) return null;
    while (true) {
      const p = peek();
      if (p && p.t === "OP" && (p.v === "*" || p.v === "/" || p.v === "%")) {
        const op = p.v;
        eat();
        const right = parseFactor();
        if (right === null) return null;
        if (op === "*") left = left * right;
        else if (op === "/") {
          if (right === 0) return null;
          left = left / right;
        } else {
          if (right === 0) return null;
          left = left % right;
        }
      } else break;
    }
    return left;
  };

  // factor = base ('^' factor)?    right-assoc
  const parseFactor = (): number | null => {
    const left = parseBase();
    if (left === null) return null;
    const p = peek();
    if (p && p.t === "OP" && p.v === "^") {
      eat();
      const right = parseFactor();
      if (right === null) return null;
      return Math.pow(left, right);
    }
    return left;
  };

  // base = '-' base | '+' base | atom
  const parseBase = (): number | null => {
    const p = peek();
    if (p && p.t === "OP" && p.v === "-") {
      eat();
      const v = parseBase();
      return v === null ? null : -v;
    }
    if (p && p.t === "OP" && p.v === "+") {
      eat();
      return parseBase();
    }
    return parseAtom();
  };

  // atom = NUM | '(' expr ')'
  const parseAtom = (): number | null => {
    const p = peek();
    if (!p) return null;
    if (p.t === "NUM") {
      eat();
      return p.v;
    }
    if (p.t === "LP") {
      eat();
      const v = parseExpr();
      if (v === null) return null;
      const q = peek();
      if (!q || q.t !== "RP") return null;
      eat();
      return v;
    }
    return null;
  };

  const result = parseExpr();
  // must have consumed every token
  if (state.i !== tokens.length) return null;
  return result;
}

/**
 * Public: safely evaluate an arithmetic expression string. Returns a
 * finite number on success, null on any tokeniser / parser failure or
 * division-by-zero. Never throws.
 */
export function safeEvalArithmetic(expr: string): number | null {
  const toks = tokenise(expr);
  if (!toks || toks.length === 0) return null;
  const withImplicitMul = insertImplicitMul(toks);
  const v = evalTokens(withImplicitMul);
  if (v === null) return null;
  if (!Number.isFinite(v)) return null;
  return v;
}

// ───────────────────────────────────────────────────────────────────────────
// Utilities
// ───────────────────────────────────────────────────────────────────────────

/** Strip fenced code blocks and inline code before regex extraction. */
function stripCodeBlocks(text: string): string {
  return text.replace(/```[\s\S]*?```/g, " ").replace(/`[^`]*`/g, " ");
}

/** Compare two numbers with absolute + relative tolerance. */
function numbersMatch(a: number, b: number): boolean {
  if (a === b) return true;
  const abs = Math.abs(a - b);
  if (abs < 1e-9) return true;
  const rel = abs / Math.max(Math.abs(a), Math.abs(b), 1);
  return rel < 1e-6;
}

/** Parse a possibly comma-formatted number string. */
function parseClaimedNumber(s: string): number | null {
  const raw = s.trim().replace(/,/g, "");
  if (!/^-?\d+(?:\.\d+)?$/.test(raw)) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

// ───────────────────────────────────────────────────────────────────────────
// Detector 1 — arithmetic
// ───────────────────────────────────────────────────────────────────────────

/**
 * Match expressions like `3(5)+7=22`, `2+2 equals 4`, `100 / 4 is 25`.
 * Looks for a sequence of arithmetic-like characters followed by an
 * explicit claim-word then a number.
 */
const ARITHMETIC_RE =
  /(?<expr>(?:[-+\d\s.,*/^%()]|\*\*){2,})\s*(?:=|⇒|=>|equals?|is\s+equal\s+to|yields)\s*(?<claimed>-?\d+(?:,\d{3})*(?:\.\d+)?)/gi;

interface ArithmeticMatch {
  expr: string;
  claimedRaw: string;
  rawMatch: string;
}

function detectArithmetic(answer: string): ArithmeticMatch[] {
  const cleaned = stripCodeBlocks(answer);
  const out: ArithmeticMatch[] = [];
  ARITHMETIC_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = ARITHMETIC_RE.exec(cleaned)) !== null) {
    const expr = (m.groups?.expr ?? "").trim();
    const claimedRaw = (m.groups?.claimed ?? "").trim();
    if (!expr || !claimedRaw) continue;
    // Expression must contain at least one operator AND at least one digit
    if (!/[-+*/%^]/.test(expr) || !/\d/.test(expr)) continue;
    // Drop pathological spans (too long — likely a paragraph slipped in)
    if (expr.length > 120) continue;
    // Normalise Python's ** to ^
    const normExpr = expr.replace(/\*\*/g, "^");
    out.push({ expr: normExpr, claimedRaw, rawMatch: m[0] });
  }
  return out;
}

// ───────────────────────────────────────────────────────────────────────────
// Detector 2 — range / list-comprehension enumeration
// ───────────────────────────────────────────────────────────────────────────

/**
 * Match `range(N)`, `range(a, b)`, `range(a, b, step)` claims plus an
 * asserted list literal. Also handle the three simplest list-comprehension
 * transforms: `x*x`, `x**2`, `x+k`, `x*k`.
 */
const RANGE_RE =
  /range\(\s*(?<a>-?\d+)(?:\s*,\s*(?<b>-?\d+))?(?:\s*,\s*(?<c>-?\d+))?\s*\)/i;

const ENUM_CLAIM_RE =
  /(?:produces|returns|outputs?|gives|is|yields|evaluates\s+to|results\s+in|prints|prints\s+out)\s*\[(?<items>[^\]]*)\]/i;

// Optional transform wrapper: [x*x for x in range(5)]
const COMP_RE =
  /\[\s*(?<expr>[a-z]\s*(?:\*\*|\*|\+|-)\s*[a-z\d]+)\s+for\s+(?<var>[a-z])\s+in\s+range\(\s*(?<a>-?\d+)(?:\s*,\s*(?<b>-?\d+))?(?:\s*,\s*(?<c>-?\d+))?\s*\)\s*\]/i;

interface EnumerationMatch {
  expectedList: number[];
  claimedList: number[];
  rawMatch: string;
}

function computeRange(a: number, b: number | null, c: number | null): number[] {
  const start = b !== null ? a : 0;
  const stop = b !== null ? b : a;
  const step = c !== null ? c : 1;
  if (step === 0) return [];
  const out: number[] = [];
  if (step > 0) {
    for (let i = start; i < stop; i += step) out.push(i);
  } else {
    for (let i = start; i > stop; i += step) out.push(i);
  }
  // cap at 200 to prevent runaway
  return out.slice(0, 200);
}

function applyListCompTransform(
  vals: number[],
  exprText: string,
  varName: string
): number[] | null {
  // Very narrow grammar: x*x, x**2, x+k, x*k, x-k, x/k
  const e = exprText.replace(/\s+/g, "");
  if (e === `${varName}*${varName}` || e === `${varName}**2`) {
    return vals.map((v) => v * v);
  }
  const m = /^([a-z])([-+*/])(-?\d+(?:\.\d+)?)$/.exec(e);
  if (m && m[1] === varName) {
    const k = Number(m[3]);
    switch (m[2]) {
      case "+":
        return vals.map((v) => v + k);
      case "-":
        return vals.map((v) => v - k);
      case "*":
        return vals.map((v) => v * k);
      case "/":
        if (k === 0) return null;
        return vals.map((v) => v / k);
    }
  }
  return null;
}

function parseClaimedList(items: string): number[] | null {
  if (!items.trim()) return [];
  const parts = items.split(",").map((s) => s.trim());
  const nums: number[] = [];
  for (const p of parts) {
    const n = parseClaimedNumber(p);
    if (n === null) return null;
    nums.push(n);
  }
  return nums;
}

function detectEnumeration(answer: string): EnumerationMatch[] {
  const cleaned = stripCodeBlocks(answer);
  const out: EnumerationMatch[] = [];

  // Pattern A: [x*x for x in range(5)] = [0, 1, 4, 9, 16]
  COMP_RE.lastIndex = 0;
  const compMatches = Array.from(cleaned.matchAll(new RegExp(COMP_RE.source, "gi")));
  for (const m of compMatches) {
    const a = Number(m.groups!.a);
    const b = m.groups!.b !== undefined ? Number(m.groups!.b) : null;
    const c = m.groups!.c !== undefined ? Number(m.groups!.c) : null;
    const baseRange = computeRange(a, b, c);
    const expected = applyListCompTransform(baseRange, m.groups!.expr!, m.groups!.var!);
    if (!expected) continue;
    // Look for a claimed list near this match (within 120 chars after)
    const tail = cleaned.slice(m.index! + m[0].length, m.index! + m[0].length + 200);
    const claimMatch = ENUM_CLAIM_RE.exec(tail);
    if (!claimMatch) continue;
    const claimed = parseClaimedList(claimMatch.groups!.items!);
    if (!claimed) continue;
    out.push({
      expectedList: expected,
      claimedList: claimed,
      rawMatch: m[0] + "..." + claimMatch[0],
    });
  }

  // Pattern B: range(N) directly asserted to equal [list]
  // Only fire if we didn't already match a comprehension overlapping this span.
  const rangeOnlyRe = new RegExp(
    RANGE_RE.source + "\\s*(?:produces|returns|outputs|gives|is|=)\\s*\\[(?<items>[^\\]]*)\\]",
    "gi"
  );
  let m2: RegExpExecArray | null;
  while ((m2 = rangeOnlyRe.exec(cleaned)) !== null) {
    const a = Number(m2.groups!.a);
    const b = m2.groups!.b !== undefined ? Number(m2.groups!.b) : null;
    const c = m2.groups!.c !== undefined ? Number(m2.groups!.c) : null;
    const expected = computeRange(a, b, c);
    const claimed = parseClaimedList(m2.groups!.items!);
    if (!claimed) continue;
    out.push({
      expectedList: expected,
      claimedList: claimed,
      rawMatch: m2[0],
    });
  }

  return out;
}

function listsMatch(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!numbersMatch(a[i]!, b[i]!)) return false;
  }
  return true;
}

// ───────────────────────────────────────────────────────────────────────────
// Detector 3 — leap-year day counts
// ───────────────────────────────────────────────────────────────────────────

/** Standard Gregorian leap-year test. */
function isLeapYear(y: number): boolean {
  return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
}

// "2024 is a leap year so it has 365 days"  — mismatch (should be 366)
// "1900 is a leap year"  — mismatch (1900 is NOT leap — century non-divisible-by-400)
const LEAP_DAYS_RE =
  /(?<year>\b(?:19|20|21)\d{2}\b)[^.]{0,80}?(?:leap\s+year)[^.]{0,80}?(?:has|it\s+has|contains)?\s*(?<days>\d{3})\s*days/gi;

interface LeapMatch {
  year: number;
  claimedDays: number;
  actualIsLeap: boolean;
  actualDays: number;
  rawMatch: string;
}

function detectLeapYear(answer: string): LeapMatch[] {
  const cleaned = stripCodeBlocks(answer);
  const out: LeapMatch[] = [];
  LEAP_DAYS_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = LEAP_DAYS_RE.exec(cleaned)) !== null) {
    const year = Number(m.groups!.year!);
    const claimedDays = Number(m.groups!.days!);
    const actualIsLeap = isLeapYear(year);
    const actualDays = actualIsLeap ? 366 : 365;
    out.push({
      year,
      claimedDays,
      actualIsLeap,
      actualDays,
      rawMatch: m[0],
    });
  }
  return out;
}

// ───────────────────────────────────────────────────────────────────────────
// Detector 4 — well-known physical constants with gross unit errors
// ───────────────────────────────────────────────────────────────────────────

interface ConstantSpec {
  name: string;
  // Regex that captures <value> and <unit>
  re: RegExp;
  // Expected value in the canonical SI unit (e.g. m/s for speed of light)
  expectedCanonical: number;
  canonicalUnit: string;
  // Function to convert the matched value+unit to canonical; null if unit unsupported
  toCanonical: (value: number, unit: string) => number | null;
  // Tolerance (fractional)
  tolerance: number;
}

const UNIT_CONSTANTS: ConstantSpec[] = [
  {
    name: "speed of light",
    re: /speed\s+of\s+light(?:[^.]*?)(?:is|=|equals|of)\s*(?<value>\d+(?:,\d{3})*(?:\.\d+)?)\s*(?<unit>km\/h|km\/hour|km\/hr|km\/s|km\/sec|m\/s|mi\/s)/gi,
    expectedCanonical: 299_792_458, // m/s
    canonicalUnit: "m/s",
    toCanonical: (value, unit) => {
      const u = unit.toLowerCase().replace(/\s+/g, "");
      switch (u) {
        case "m/s":
          return value;
        case "km/s":
        case "km/sec":
          return value * 1000;
        case "km/h":
        case "km/hour":
        case "km/hr":
          return (value * 1000) / 3600;
        case "mi/s":
          return value * 1609.344;
      }
      return null;
    },
    tolerance: 0.05,
  },
];

interface UnitMatch {
  name: string;
  claimedValue: number;
  claimedUnit: string;
  canonicalClaim: number;
  expectedCanonical: number;
  canonicalUnit: string;
  rawMatch: string;
}

function detectUnitConstants(answer: string): UnitMatch[] {
  const cleaned = stripCodeBlocks(answer);
  const out: UnitMatch[] = [];
  for (const spec of UNIT_CONSTANTS) {
    spec.re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = spec.re.exec(cleaned)) !== null) {
      const valStr = (m.groups?.value ?? "").replace(/,/g, "");
      const unit = m.groups?.unit ?? "";
      const claimedValue = Number(valStr);
      if (!Number.isFinite(claimedValue)) continue;
      const canonicalClaim = spec.toCanonical(claimedValue, unit);
      if (canonicalClaim === null) continue;
      out.push({
        name: spec.name,
        claimedValue,
        claimedUnit: unit,
        canonicalClaim,
        expectedCanonical: spec.expectedCanonical,
        canonicalUnit: spec.canonicalUnit,
        rawMatch: m[0],
      });
    }
  }
  return out;
}

// ───────────────────────────────────────────────────────────────────────────
// Detector 5 — linear equations `ax + b = c` and their solution steps
// ───────────────────────────────────────────────────────────────────────────

/**
 * Parse an equation in LHS-first form: `ax + b = c` (or `ax - b = c`).
 * Coefficient is optional (default 1), variable is a single letter. Out of
 * scope: RHS-first (`c = ax + b`), variable-on-both-sides, reordered LHS
 * (`b + ax = c`), and any non-linear terms.
 */
const LINEAR_EQ_RE =
  /(?<!\w)(?<coeffSign>-?)(?<coeffDigits>\d+(?:\.\d+)?)?\s*(?<var>[a-z])(?![a-z])\s*(?<bSign>[+\-])\s*(?<b>\d+(?:\.\d+)?)\s*=\s*(?<c>-?\d+(?:\.\d+)?)/gi;

/**
 * Parse a single-variable claim like `x = 5`, `3x = 15`, or `-2y = -10`.
 * Used to find the answer's asserted solution and intermediate steps.
 */
const VAR_CLAIM_RE =
  /(?<!\w)(?<coeffSign>-?)(?<coeffDigits>\d+(?:\.\d+)?)?\s*(?<var>[a-z])(?![a-z])\s*=\s*(?<value>-?\d+(?:\.\d+)?)/gi;

interface LinearEqSolution {
  variable: string; // lowercased for map-key consistency
  coefficient: number;
  b: number;
  c: number;
  solution: number;
}

/**
 * Solve any linear equations in (question + answer) and emit a verification
 * for every `kx = v` claim in the answer. Captured `expr_text` is the
 * literal claim text so the aggregator's substring suppression rule can
 * match it against NLI contradiction claims like
 * "Divide by 3: x = 5." — the direct driver of the `subtle-math` failure.
 */
function detectLinearEquations(
  question: string,
  answer: string
): { verifications: RecomputeVerification[]; mismatches: RecomputeResult["mismatches"] } {
  const verifications: RecomputeVerification[] = [];
  const mismatches: RecomputeResult["mismatches"] = [];

  const cleanedA = stripCodeBlocks(answer);
  const source = stripCodeBlocks(question) + "\n" + cleanedA;

  const solvedByVar = new Map<string, LinearEqSolution>();
  LINEAR_EQ_RE.lastIndex = 0;
  let em: RegExpExecArray | null;
  while ((em = LINEAR_EQ_RE.exec(source)) !== null) {
    const sign = em.groups?.coeffSign ?? "";
    const digits = em.groups?.coeffDigits;
    const coefficient = (sign === "-" ? -1 : 1) * (digits ? Number(digits) : 1);
    if (!Number.isFinite(coefficient) || coefficient === 0) continue;
    const variable = (em.groups?.var ?? "").toLowerCase();
    if (!variable) continue;
    const b = (em.groups?.bSign === "-" ? -1 : 1) * Number(em.groups?.b);
    const c = Number(em.groups?.c);
    if (!Number.isFinite(b) || !Number.isFinite(c)) continue;
    const solution = (c - b) / coefficient;
    if (!Number.isFinite(solution)) continue;
    if (!solvedByVar.has(variable)) {
      solvedByVar.set(variable, { variable, coefficient, b, c, solution });
    }
  }

  if (solvedByVar.size === 0) return { verifications, mismatches };

  VAR_CLAIM_RE.lastIndex = 0;
  let cm: RegExpExecArray | null;
  while ((cm = VAR_CLAIM_RE.exec(cleanedA)) !== null) {
    const variable = (cm.groups?.var ?? "").toLowerCase();
    const eq = solvedByVar.get(variable);
    if (!eq) continue;
    const sign = cm.groups?.coeffSign ?? "";
    const digits = cm.groups?.coeffDigits;
    const claimCoeff = (sign === "-" ? -1 : 1) * (digits ? Number(digits) : 1);
    if (!Number.isFinite(claimCoeff) || claimCoeff === 0) continue;
    const value = Number(cm.groups?.value);
    if (!Number.isFinite(value)) continue;
    const expected = claimCoeff * eq.solution;
    const matches = numbersMatch(value, expected);
    const exprText = cm[0].trim();
    verifications.push({
      kind: "linear-equation",
      expr_text: exprText,
      claimed: String(value),
      computed: String(expected),
      matches,
      confidence: 1.0,
    });
    if (!matches) {
      mismatches.push({
        kind: "linear-equation",
        expr_text: exprText,
        claimed: String(value),
        computed: String(expected),
      });
    }
  }

  return { verifications, mismatches };
}

// ───────────────────────────────────────────────────────────────────────────
// Public: run the whole recompute pass on an answer
// ───────────────────────────────────────────────────────────────────────────

export async function runRecomputePass(
  answer: string,
  question?: string
): Promise<RecomputeResult> {
  const t0 = Date.now();

  const verifications: RecomputeVerification[] = [];
  const mismatches: RecomputeResult["mismatches"] = [];

  // For enumeration detection, concatenate question + answer: a list
  // comprehension is often asserted by the question, then the answer
  // claims what it "prints". See python-list-comp test case.
  const enumerationScope =
    question && question.trim().length > 0
      ? question.trim() + "\n" + answer
      : answer;

  // --- arithmetic
  for (const a of detectArithmetic(answer)) {
    const claimed = parseClaimedNumber(a.claimedRaw);
    if (claimed === null) continue;
    const computed = safeEvalArithmetic(a.expr);
    if (computed === null) continue;
    const matches = numbersMatch(computed, claimed);
    const v: RecomputeVerification = {
      kind: "arithmetic",
      expr_text: a.expr,
      claimed: String(claimed),
      computed: String(computed),
      matches,
      confidence: 1.0,
    };
    verifications.push(v);
    if (!matches) {
      mismatches.push({
        kind: "arithmetic",
        expr_text: a.expr,
        claimed: String(claimed),
        computed: String(computed),
      });
    }
  }

  // --- enumeration (uses question+answer scope)
  for (const e of detectEnumeration(enumerationScope)) {
    const matches = listsMatch(e.expectedList, e.claimedList);
    const v: RecomputeVerification = {
      kind: "enumeration",
      expr_text: e.rawMatch.slice(0, 80),
      claimed: "[" + e.claimedList.join(",") + "]",
      computed: "[" + e.expectedList.join(",") + "]",
      matches,
      confidence: 1.0,
    };
    verifications.push(v);
    if (!matches) {
      mismatches.push({
        kind: "enumeration",
        expr_text: e.rawMatch.slice(0, 80),
        claimed: "[" + e.claimedList.join(",") + "]",
        computed: "[" + e.expectedList.join(",") + "]",
      });
    }
  }

  // --- leap year
  for (const ly of detectLeapYear(answer)) {
    const matches = ly.claimedDays === ly.actualDays;
    const v: RecomputeVerification = {
      kind: "leap-year",
      expr_text: `${ly.year} is ${ly.actualIsLeap ? "" : "not "}a leap year`,
      claimed: `${ly.claimedDays} days`,
      computed: `${ly.actualDays} days`,
      matches,
      confidence: 1.0,
    };
    verifications.push(v);
    if (!matches) {
      mismatches.push({
        kind: "leap-year",
        expr_text: v.expr_text,
        claimed: v.claimed,
        computed: v.computed,
      });
    }
  }

  // --- unit constants
  for (const u of detectUnitConstants(answer)) {
    const tolerance = 0.05;
    const rel =
      Math.abs(u.canonicalClaim - u.expectedCanonical) /
      Math.max(Math.abs(u.expectedCanonical), 1);
    const matches = rel < tolerance;
    const v: RecomputeVerification = {
      kind: "unit",
      expr_text: `${u.name} = ${u.claimedValue} ${u.claimedUnit}`,
      claimed: `${u.canonicalClaim.toPrecision(4)} ${u.canonicalUnit}`,
      computed: `${u.expectedCanonical.toPrecision(4)} ${u.canonicalUnit}`,
      matches,
      confidence: 1.0,
    };
    verifications.push(v);
    if (!matches) {
      mismatches.push({
        kind: "unit",
        expr_text: v.expr_text,
        claimed: v.claimed,
        computed: v.computed,
      });
    }
  }

  // --- linear equations (uses question + answer scope)
  {
    const { verifications: ve, mismatches: mi } = detectLinearEquations(
      question ?? "",
      answer
    );
    verifications.push(...ve);
    mismatches.push(...mi);
  }

  const latency_ms = Date.now() - t0;
  const notes =
    verifications.length === 0
      ? "No arithmetic / enumeration / leap-year / unit / linear-equation claims detected."
      : `Recomputed ${verifications.length} claim(s); ${mismatches.length} mismatch(es).`;

  return {
    ran: true,
    expressions_found: verifications.length,
    verifications,
    mismatches,
    notes,
    latency_ms,
  };
}
