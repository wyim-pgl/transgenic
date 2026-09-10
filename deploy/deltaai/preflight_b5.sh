#!/bin/bash
# A40 preflight (#69 계열): 첫 optimizer step 이전에 학습이 실제로 무엇을 먹는지 확인한다.
#
# 이 스크립트가 존재하는 이유는 A28이다 — 레시피가 바뀌었는데 폐기된 데이터/tokenizer 사슬이
# 조용히 A40 런이 되는 것을 막는다. 기대값은 하드코딩하지 않고 **freeze JSON에서 읽는다**:
# 하드코딩하면 freeze와 스크립트가 따로 낡는다.
#
#   preflight_b5.sh --bundle <training_input> --repo <transgenic repo> [--freeze <json>]
#
# exit 0 전부 통과 · exit 3 검사 실패 · exit 4 검사할 도구가 없음(모름을 통과로 바꾸지 않는다)
set -uo pipefail

BUNDLE=""; REPO=""; FREEZE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2;;
    --repo)   REPO="$2"; shift 2;;
    --freeze) FREEZE="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$BUNDLE" ] && [ -n "$REPO" ] || { echo "usage: $0 --bundle <dir> --repo <dir> [--freeze <json>]" >&2; exit 2; }
[ -d "$BUNDLE" ] || { echo "FAIL bundle: $BUNDLE is not a directory" >&2; exit 3; }
[ -d "$REPO" ]   || { echo "FAIL repo: $REPO is not a directory" >&2; exit 3; }

PY=$(command -v python3 || command -v python) || { echo "FAIL tools: no python (cannot verify; refusing)" >&2; exit 4; }
"$PY" -c 'import duckdb' 2>/dev/null || { echo "FAIL tools: python has no duckdb (cannot verify; refusing)" >&2; exit 4; }

if [ -z "$FREEZE" ]; then
  FREEZE=$(ls "$BUNDLE"/freeze/*a40*.freeze.json 2>/dev/null | head -1)
fi
[ -n "$FREEZE" ] && [ -f "$FREEZE" ] || { echo "FAIL freeze: no A40 freeze record under $BUNDLE/freeze (pass --freeze)" >&2; exit 3; }

FAIL=0
ok()   { printf 'PASS  %s\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1" >&2; FAIL=1; }

# ---- freeze가 말하는 기대값 --------------------------------------------------
read -r EXP_MD5 EXP_SHA EXP_ROWS EXP_BYTES DB_BASE < <(
  "$PY" - "$FREEZE" <<'PYX'
import json, os, sys
d = json.load(open(sys.argv[1]))
print(d["file_md5"], d["file_sha256"], d["geneList_rows"], d["file_bytes"], os.path.basename(d["database"]))
PYX
) || { echo "FAIL freeze: cannot read expected values from $FREEZE" >&2; exit 3; }
echo "freeze: $FREEZE -> $DB_BASE  rows=$EXP_ROWS  bytes=$EXP_BYTES"

# ---- 1. 번들 DB가 freeze가 지목한 그 객체인가 --------------------------------
DB="$BUNDLE/db/$DB_BASE"
if [ ! -f "$DB" ]; then
  bad "db: $DB is missing (the bundle does not carry the object the freeze names)"
else
  GOT_BYTES=$(stat -c %s "$DB")
  [ "$GOT_BYTES" = "$EXP_BYTES" ] && ok "db bytes $GOT_BYTES" || bad "db bytes $GOT_BYTES != $EXP_BYTES"
  GOT_MD5=$(md5sum "$DB" | cut -d' ' -f1)
  [ "$GOT_MD5" = "$EXP_MD5" ] && ok "db md5 $GOT_MD5" || bad "db md5 $GOT_MD5 != $EXP_MD5"
  GOT_SHA=$(sha256sum "$DB" | cut -d' ' -f1)
  [ "$GOT_SHA" = "$EXP_SHA" ] && ok "db sha256 $GOT_SHA" || bad "db sha256 $GOT_SHA != $EXP_SHA"
  GOT_ROWS=$("$PY" -c "import duckdb,sys;print(duckdb.connect(sys.argv[1],read_only=True).execute('SELECT count(*) FROM geneList').fetchone()[0])" "$DB" 2>/dev/null)
  [ "$GOT_ROWS" = "$EXP_ROWS" ] && ok "geneList rows $GOT_ROWS" || bad "geneList rows ${GOT_ROWS:-unreadable} != $EXP_ROWS"
fi

# ---- 2. 폐기 코퍼스가 같이 실려 있지 않은가 ----------------------------------
STALE=$(ls "$BUNDLE"/db/*.db 2>/dev/null | grep -v "/$DB_BASE\$" || true)
[ -z "$STALE" ] && ok "db dir carries only the A40 object" \
  || bad "db dir also carries superseded databases (A28: a stale chain must not become an A40 run):
$STALE"

# ---- 3. 번들이 ACCESS에서 성립하는가 (dangling 링크 없음) --------------------
# repo/HFmodels is the HuggingFace cache: its snapshots/ -> blobs/ links are relative and intentional, and it
# lives inside the bundle only because the repo is bind-mounted there. Not a transfer defect.
DANGLING=$(find "$BUNDLE" -type l -not -path "*/HFmodels/*" ! -exec test -e {} \; -print 2>/dev/null)
[ -z "$DANGLING" ] && ok "no dangling symlinks in the bundle" || bad "dangling symlinks:
$DANGLING"
LINKS=$(find "$BUNDLE" -type l -not -path "*/HFmodels/*" | wc -l)
[ "$LINKS" -eq 0 ] && ok "bundle is self-contained (0 symlinks)" \
  || echo "NOTE  $LINKS symlink(s) still present — the transfer MUST materialise them (rsync -L). intent.md R2."

# ---- 4. tokenizer가 3토큰 빈 타깃을 낸다 (#58 / A40) -------------------------
# 실제로 토큰화해서 길이를 잰다. 소스를 grep 하면 파일을 틀리게 고르고도 결론이 나온다 —
# 이 검사 자체가 처음에 datasets.py 를 보고 "3토큰 없음"이라는 허위 실패를 냈다(quarantine.md §1g).
# stdout only, last line: torch 2.14 (NGC 26.08) prints an import-time warning on stderr
# ("KernelPreference is an Enum subclass ... torch.compile") that turned a THREE into "could not verify"
# and refused the launch (rehearsal 3124100, 2026-09-10). The check's own reasons are printed on stdout.
TOK=$("$PY" - "$REPO" <<'PYX' 2>/dev/null | tail -n 1
import pathlib, sys
repo = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(repo / "src"))
# The class is GFFTokenizer and <empty> exists only in the v3 vocabulary (A26). The earlier text imported
# a name that does not exist, so the tokenised check had never run anywhere and every PASS was the source
# scan below, mislabelled "import unavailable". Measured on the GB10 SIF, 2026-09-08.
reason = ""
try:
    from transgenic.model.tokenization_transgenic import GFFTokenizer as T
    toks = T(vocab_version="v3").tokenize("<empty>")
    print("THREE" if list(toks) == ["<s>", "<empty>", "</s>"] else f"WRONG {toks}")
    raise SystemExit(0)
except SystemExit:
    raise
except Exception as e:                       # torch absent, or a real defect: say which
    reason = f"{type(e).__name__}: {e}"[:160]
# 임포트가 안 되는 환경(torch 부재 등)에서는 정본 두 파일의 선언으로 낮춘 검사를 하고 그 이유를 말한다.
tok = repo / "src" / "transgenic" / "model" / "tokenization_transgenic.py"
con = repo / "src" / "transgenic" / "utils" / "gsf_contract.py"
if not (tok.is_file() and con.is_file()):
    print("UNREADABLE tokenizer/contract source missing"); raise SystemExit(0)
a = '["<s>", "<empty>", "</s>"]' in tok.read_text().replace("'", '"')
b = "return 3" in con.read_text()
print(("THREE_SOURCE " + reason) if (a and b) else ("MISSING " + reason))
PYX
)
case "$TOK" in
  THREE)        ok "tokenizer emits exactly <s> <empty> </s> (tokenised, #58 / A40)";;
  THREE_SOURCE*) ok "tokenizer declares <s> <empty> </s> (SOURCE CHECK ONLY - tokenised check failed: ${TOK#THREE_SOURCE }; #58 / A40)";;
  WRONG*)       bad "tokenizer emits $TOK, not three tokens (#58 / A40)";;
  MISSING*)     bad "tokenizer: the three-token empty target is not declared (#58 / A40)";;
  *)            bad "tokenizer: could not verify - $TOK";;
esac

# ---- 4b. encoder cache present (no network at job start) ----------------------
# train_HyenaTransgenic.py sets HF_HOME=./HFmodels; the weights must already be under the repo or the
# job downloads at step 0 -- six concurrent jobs into one directory, on a node that may have no route out.
ENC=$(sed -n 's/.*"encoder_model": *"\([^"]*\)".*/\1/p' "$REPO/configs/b5_400m_win_v3.json" | head -1)
ENCDIR="$REPO/HFmodels/hub/models--${ENC//\//--}"
if [ -n "$ENC" ] && ls "$ENCDIR"/snapshots/*/model.safetensors >/dev/null 2>&1 \
   && ls "$REPO/HFmodels/models--${ENC//\//--}"/snapshots/*/tokenizer_config.json >/dev/null 2>&1; then
  ok "encoder cache present for $ENC (weights + tokenizer under repo/HFmodels; offline load)"
else
  bad "encoder cache missing for ${ENC:-<unparsed>}: expected $ENCDIR/snapshots/*/model.safetensors and the tokenizer
      under repo/HFmodels/models--...; the job would download at step 0. Ship the cache first."
fi

# ---- 5. seedable sampler (#59) ----------------------------------------------
if grep -rq "use_seedable_sampler" "$REPO/train" 2>/dev/null; then
  ok "trainer uses a seedable sampler (#59)"
else
  bad "trainer has no use_seedable_sampler: a mid-epoch resume will replay a different permutation (#59).
      NOTE: 확인 전에 저장소를 찍을 것 — $(cd "$REPO" && git log --oneline -1 2>/dev/null || echo 'not a git repo')"
fi

# ---- 6. 정본 레시피 ----------------------------------------------------------
[ -f "$BUNDLE/configs/b5_400m_win_v3.json" ] && ok "canonical recipe b5_400m_win_v3.json present" \
  || bad "configs/b5_400m_win_v3.json missing (A26 canonical recipe)"

echo
if [ "$FAIL" -eq 0 ]; then echo "PREFLIGHT PASS — this run would consume the A40 object the freeze names"; exit 0; fi
echo "PREFLIGHT FAIL — do not launch" >&2; exit 3
